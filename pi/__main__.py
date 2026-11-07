"""Allow ``python -m pi`` from the pi-python directory."""

import sys


def main():
    if len(sys.argv) > 1:
        if sys.argv[1] == "durable":
            from .durable.cli import main as durable_main

            return durable_main(sys.argv[2:])
        if sys.argv[1] == "durable-worker":
            from .durable.worker import main as worker_main

            return worker_main(sys.argv[2:])
        if sys.argv[1] == "server":
            from .durable.server import main as server_main

            return server_main(sys.argv[2:])
    from .coding_agent.cli import main as cli_main

    return cli_main()


if __name__ == "__main__":
    sys.exit(main())
