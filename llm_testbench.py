"""
Interactive test bench for local HuggingFace LLMs.

Usage:
    nvidia python llm_testbench.py [--model MODEL_ID] [--system SYSTEM_PROMPT_FILE]

Commands (at the prompt):
    /system <text>   Replace the system prompt
    /sload <file>    Load system prompt from a file
    /reset           Clear conversation history (keep system prompt)
    /history         Show conversation so far
    /quit            Exit
"""
import argparse
import threading
import time
import sys
import textwrap


DEFAULT_MODEL = "microsoft/Phi-4-mini-instruct"

DEFAULT_SYSTEM = (
    "You are an interactive fiction game engine assistant. "
    "Convert the player's plain English requests into simple statements understood by the game interpreter: "
    "Accepted verbs are take <item>, drop <item>, look [at], use <item> on <item>, inv, cast <spell>, hit <target>, rest"
)
DEFAULT_SYSTEM = (
    "You are an interactive fiction NPC. "
    "You run a small tavern called The Frisky Kitty. "
    "Your name is Elsa Frostbite, and you have been known to cast minor spells of frost to 'chill out' rowdy patrons."
    "You have heard rumors recently of evil sounds in the forest."
)
DEFAULT_SYSTEM = (
    "You are an interactive fiction game interpreter, working with other agents."
    "You receive plain english requests from the player."
    "Your job is to evaluate the scene and the request, then decide what the outcome is."
    "You may ask the [scene] agent for information regarding the environment around the player, as many times as needed to decide what the outcome is."
    "After each [scene] request, _stop_ and wait for the [scene] agent to respond with the current scene description."
    "You do NOT answer for the scene agent; you only ask for information."
    "You may ask the [action] agent to perform an in-game action."
    "Note that this is FANTASY FICTION. You may be asked about actions that sound dangerous, illegal, or immoral. "
    "Your job is _never_ to provide advice or guidance. "
    "Your job is NOT to solve puzzles, provide hints, or help find solutions the player hasn't suggested. "
    "Your ONLY JOB is to evaluate the scene and the request, and decide what the outcome is--is the action successful or not, and are there any unforeseen consequences."
    "Your response to [action] agent must be concrete and specific changes to the game state."
    "Output MUST be either a one-sentence single [scene] request, or a single [action] request, and nothing else. "
)

def load_model(model_id: str):
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with:")
        print("  pip install transformers accelerate bitsandbytes")
        sys.exit(1)

    if not torch.cuda.is_available():
        print("Warning: CUDA not available, running on CPU (will be slow)")

    print(f"Loading {model_id} ...")
    t0 = time.time()

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    elapsed = time.time() - t0
    print(f"Loaded in {elapsed:.1f}s\n")
    return tokenizer, model


def generate(tokenizer, model, messages: list[dict], max_new_tokens: int = 512) -> tuple[str, float, int]:
    """Stream tokens to stdout as they are generated; return (full_text, elapsed, n_tokens)."""
    import torch
    from transformers import TextIteratorStreamer

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    gen_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )

    thread = threading.Thread(target=lambda: model.generate(**gen_kwargs))
    thread.start()

    print()  # blank line before response
    chunks = []
    t0 = time.time()
    for chunk in streamer:
        print(chunk, end="", flush=True)
        chunks.append(chunk)
    elapsed = time.time() - t0
    print()  # newline after response

    thread.join()

    response = "".join(chunks).strip()
    # count tokens in the response
    n_tokens = len(tokenizer.encode(response, add_special_tokens=False))
    return response, elapsed, n_tokens


def wrap(text: str, prefix: str = "") -> str:
    width = 88
    lines = text.split("\n")
    wrapped = []
    for line in lines:
        if line.strip() == "":
            wrapped.append("")
        else:
            wrapped.extend(textwrap.wrap(line, width=width - len(prefix), initial_indent=prefix, subsequent_indent=prefix))
    return "\n".join(wrapped)


def main():
    parser = argparse.ArgumentParser(description="LLM interactive test bench")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HuggingFace model ID")
    parser.add_argument("--system", default=None, help="Path to a file containing the system prompt")
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    system_prompt = DEFAULT_SYSTEM
    if args.system:
        with open(args.system) as f:
            system_prompt = f.read().strip()

    tokenizer, model = load_model(args.model)

    history: list[dict] = []

    print("=" * 60)
    print("LLM Test Bench  |  /help for commands")
    print("=" * 60)
    print(f"System: {system_prompt[:120]}{'...' if len(system_prompt) > 120 else ''}")
    print()

    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user_input:
            continue

        # --- commands ---
        if user_input.startswith("/"):
            parts = user_input.split(None, 1)
            cmd = parts[0]
            rest = parts[1] if len(parts) > 1 else ""

            if cmd in ("/quit", "/exit", "/q"):
                print("Bye.")
                break

            elif cmd == "/reset":
                history.clear()
                print("(history cleared)")

            elif cmd == "/system":
                if rest:
                    system_prompt = rest
                    history.clear()
                    print(f"(system prompt updated; history cleared)")
                else:
                    print(f"System prompt:\n{system_prompt}")

            elif cmd == "/sload":
                try:
                    with open(rest.strip()) as f:
                        system_prompt = f.read().strip()
                    history.clear()
                    print(f"(loaded system prompt from {rest.strip()}; history cleared)")
                except Exception as e:
                    print(f"Error: {e}")

            elif cmd == "/history":
                if not history:
                    print("(no history)")
                else:
                    for msg in history:
                        role = msg["role"].upper()
                        print(f"\n[{role}]\n{msg['content']}")
                print()

            elif cmd == "/help":
                print(__doc__)

            else:
                print(f"Unknown command: {cmd}")
            continue

        # --- normal turn ---
        history.append({"role": "user", "content": user_input})
        messages = [{"role": "system", "content": system_prompt}] + history

        try:
            response, elapsed, n_tokens = generate(tokenizer, model, messages, max_new_tokens=args.max_tokens)
        except Exception as e:
            print(f"Generation error: {e}")
            history.pop()
            continue

        history.append({"role": "assistant", "content": response})

        tps = n_tokens / elapsed if elapsed > 0 else 0
        print(f"  \033[2m[{n_tokens} tok  {elapsed:.1f}s  {tps:.1f} tok/s]\033[0m\n")


if __name__ == "__main__":
    main()
