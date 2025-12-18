! pip install mistralai
!pip install transformers datasets torch
from transformers import AutoTokenizer, AutoModelWithLMHead
import pandas as pd
from datasets import load_dataset, Dataset
import json
import gzip
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import requests
from collections import defaultdict
from mistralai import Mistral
import os

!wget https://raw.githubusercontent.com/interactive-fiction-class/interactive-fiction-class-data/master/light_dialogue/light_dialogue_data_train.json.gz -O light_dialogue_data_train.json.gz
!gunzip light_dialogue_data_train.json.gz
light_dialogue_json_filename = 'light_dialogue_data_train.json'
f = open(light_dialogue_json_filename)
light_dialogues = json.load(f)

def get_dialogue_messages(dialogue):
    """
    Constructs a structured list of dialogue messages for Mistral's format.
    """
    character_order = dialogue["character"]
    speech = dialogue["speech"]
    emotes = dialogue["emote"]
    actions = dialogue["action"]

    # First character is "user", second is "assistant"
    if len(set(character_order)) < 2:
        raise ValueError("Dialogue must involve at least two distinct characters.")

    user_character = character_order[0].lower()  # First character as user
    assistant_character = None
    for char in character_order:
        if char.lower() != user_character:
            assistant_character = char.lower()
            break

    if not assistant_character:
        raise ValueError("Dialogue must have an assistant character.")

    # Build messages
    messages = []
    for i, character in enumerate(character_order):
        role = "user" if character.lower() == user_character else "assistant"
        content = speech[i].capitalize().strip()

        # Add dialogue message
        if content:
            messages.append({"role": role, "content": content})

        if emotes[i]:
            messages.append({"role": role, "content": f"*{emotes[i].capitalize().strip()}*"})
        if actions[i]:
            messages.append({"role": role, "content": f"*{actions[i].capitalize().strip()}*"})

    return messages


def create_dialogue_finetuning_data(filename, max_dialogues=10000):
    """
    Creates a JSONL file for fine-tuning in Mistral's conversational format.
    """
    fine_tuning_data = []

    for i in range(min(max_dialogues, len(light_dialogues))):
        dialogue = light_dialogues[i]

        try:
            messages = get_dialogue_messages(dialogue)
            data = {"messages": messages}
            fine_tuning_data.append(data)
        except ValueError as e:
            print(f"Skipping dialogue {i} due to error: {e}")

    # Save to JSONL file
    with open(filename, 'w', encoding='utf-8') as out:
        for data in fine_tuning_data:
            out.write(json.dumps(data))
            out.write('\n')


# File to save the fine-tuning data
jsonl_filename = 'fine_tune_LIGHT_dialogue.jsonl'
create_dialogue_finetuning_data(jsonl_filename)

!wget https://raw.githubusercontent.com/mistralai/mistral-finetune/main/utils/reformat_data.py
!python reformat_data.py fine_tune_LIGHT_dialogue.jsonl

def split_jsonl_file(input_file, train_file, val_file, val_size):
    """
    Splits a JSONL file into training and validation sets by cutting a specific
    number of lines from the end for validation.
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    train_lines = lines[:-val_size]
    val_lines = lines[-val_size:]

    with open(train_file, 'w', encoding='utf-8') as train_f:
        train_f.writelines(train_lines)

    with open(val_file, 'w', encoding='utf-8') as val_f:
        val_f.writelines(val_lines)

# File paths
input_jsonl = "fine_tune_LIGHT_dialogue.jsonl"
train_jsonl = "fine_tune_LIGHT_dialogue_train.jsonl"
val_jsonl = "fine_tune_LIGHT_dialogue_val.jsonl"

split_jsonl_file(input_jsonl, train_jsonl, val_jsonl, val_size=50)

api_key = "API_KEY"

client = Mistral(api_key=api_key)

fine_tune_LIGHT_dialogue_train = client.files.upload(
    file={
        "file_name": "fine_tune_LIGHT_dialogue_train.jsonl",
        "content": open("fine_tune_LIGHT_dialogue_train.jsonl", "rb"),
    }
)
fine_tune_LIGHT_dialogue_val = client.files.upload(file={
    "file_name": "fine_tune_LIGHT_dialogue_val.jsonl",
    "content": open("fine_tune_LIGHT_dialogue_val.jsonl", "rb"),
})
# create a fine-tuning job
created_jobs = client.fine_tuning.jobs.create(
    model="mistral-large-latest",
    training_files=[{"file_id": fine_tune_LIGHT_dialogue_train.id, "weight": 1}],
    validation_files=[fine_tune_LIGHT_dialogue_val.id],
    hyperparameters={
        "training_steps": 15,
        "learning_rate":0.0001
    },
    auto_start=False
)

# start a fine-tuning job
client.fine_tuning.jobs.start(job_id = created_jobs.id)

created_jobs
