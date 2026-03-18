"""CLI for labsim."""
import sys, json, argparse
from .core import Labsim

def main():
    parser = argparse.ArgumentParser(description="LabSim — Virtual STEM Lab. AI-powered physics, chemistry, and biology experiment simulations.")
    parser.add_argument("command", nargs="?", default="status", choices=["status", "run", "info"])
    parser.add_argument("--input", "-i", default="")
    args = parser.parse_args()
    instance = Labsim()
    if args.command == "status":
        print(json.dumps(instance.get_stats(), indent=2))
    elif args.command == "run":
        print(json.dumps(instance.process(input=args.input or "test"), indent=2, default=str))
    elif args.command == "info":
        print(f"labsim v0.1.0 — LabSim — Virtual STEM Lab. AI-powered physics, chemistry, and biology experiment simulations.")

if __name__ == "__main__":
    main()
