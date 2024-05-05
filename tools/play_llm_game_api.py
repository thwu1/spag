import json
from tqdm import tqdm
import asyncio

import transformers

from arguments import CustomTrainingArguments

from utils import convert_game_history_to_query
from api_server import ClientGroup


def load_keyword_list(args, data_path):
    with open(data_path, "r") as f:
        keywords = f.read().strip().split("\n")
    return keywords


def get_player(args, model_name_or_path):
    return ClientGroup(8, model_name_or_path)


async def play_games(args, players, words):
    batch_games = [
        {"history": [], "target_word": keyword, "max_turns": args.taboo_max_turns}
        for keyword in words
    ]
    all_outputs = []
    gen_kwargs = {
        "max_tokens": args.max_new_tokens,
        "temperature": 1.2,
    }

    for taboo_turn in range(2 * args.taboo_max_turns):
        batch_size = len(batch_games)
        next_player = "attacker" if taboo_turn % 2 == 0 else "defender"
        model = players[next_player]

        batch_queries = [
            {
                "query": convert_game_history_to_query(
                    game["history"],
                    target_word=game["target_word"],
                    max_turns=game["max_turns"],
                ),
                "query_id": game["target_word"],
            }
            for game in batch_games
        ]

        batch_text = [item["query"] for item in batch_queries]

        output_seq = await model.run(batch_text, **gen_kwargs)

        finished_ids = []
        for idx in range(batch_size):
            response_sample = output_seq[idx]
            batch_games[idx]["history"].append(
                {"role": next_player, "content": response_sample}
            )

            if (
                "i know the word" in response_sample.lower()
                and next_player == "defender"
            ):
                # early stop to speed up inference
                all_outputs.append(batch_games[idx])
                finished_ids.append(idx)

        batch_games = [
            game for idx, game in enumerate(batch_games) if idx not in finished_ids
        ]
        if len(batch_games) == 0:
            break

    all_outputs.extend(batch_games)
    return all_outputs


async def main():
    parser = transformers.HfArgumentParser(CustomTrainingArguments)
    args = parser.parse_args_into_dataclasses()[0]

    eval_dataset = load_keyword_list(args, args.data_path)

    # setup model
    # ---------------------------------------------------------------------------------
    players = dict()
    players["attacker"] = get_player(args, args.attacker_model_name_or_path)

    if args.attacker_model_name_or_path == args.defender_model_name_or_path:
        players["defender"] = players["attacker"]
    else:
        players["defender"] = get_player(args, args.defender_model_name_or_path)

    all_outputs = []

    for i in tqdm(range(0, len(eval_dataset), args.batch_size)):
        batch_words = eval_dataset[i : i + args.batch_size]
        outputs = await play_games(args, players, batch_words)
        all_outputs.extend(outputs)

    output_path = f"{args.output_dir}/{args.model_prefix}_{args.task_type}_{args.data_suffix}.json"

    # sort according to eval_dataset order
    dic = {item["target_word"]: item for item in all_outputs}
    all_outputs = [dic[word] for word in eval_dataset]

    json.dump(all_outputs, open(output_path, "w"), indent=4)


if __name__ == "__main__":
    asyncio.run(main())
