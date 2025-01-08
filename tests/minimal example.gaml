
model minimal_example

global {

    int gama_server_port <- 0;
    bool over <- false;

    init {
        create network_manager{
            do start_server;
            do send_observations;
        }
        write "init over";
    }

    reflex communication{
        write "communicating";
        ask network_manager{
            do read_actions;
            // in between those two should be the rest of the execution of the model, but we are simplifying things here
            do send_reward;
            do send_observations;
        }
        write "communication done";
    }
}

species network_manager skills:[network] {

    action start_server {
        write "Starting communication server on port: " + gama_server_port;
        do connect to: "localhost" protocol: "tcp_client" port: gama_server_port raw: true;
    }

    action send_observations {

        if not over{
            do send to: "localhost:" + gama_server_port contents:"[1]\n";
        }
    }


    action send_reward {

        if not over{
            do send to: "localhost:" + gama_server_port contents:"123\n";
        }
    }

    action read_actions{
        if not over{
            // This loop is just to wait for messages
            loop while: !has_more_message()  {
                do fetch_message_from_network;
            }

            //This second loop will only be reached once a message has been found into the agent's mailbox
            loop while: has_more_message() {
                message s <- fetch_message();
                write "at cycle: " + cycle + ", received from server: " + s.contents;
                if s.contents contains "END" {
                    over <- true;
                    break;
                }
            }
        }
    }
}

experiment expe {

    parameter "communication_port" var:gama_server_port;

}