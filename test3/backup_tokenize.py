
# def tokenize(tokenizer, question: str, answer: str):
#     """
#     Tokenize a data element.
#     We first append the <EOS> token to the question / answer pair.
#     Then we tokenize and construct the ground truth `labels`.
#     `labels[i] == -100` for the question or masked out parts, since we only want to supervise
#     the answer.
#     """
#     full_text = f"{question} {answer}{tokenizer.eos_token}"

#     tokenizer.padding_side = "right"
#     tokenizer.pad_token = tokenizer.eos_token
#     full = tokenizer(full_text, padding="max_length", truncation=True, max_length=128)

#     input_ids = full["input_ids"]
#     question_len = len(tokenizer(question)["input_ids"])

#     # Create labels: mask out the prompt part
#     labels = [-100] * question_len + input_ids[question_len:]

#     for i in range(len(labels)):
#         if full["attention_mask"][i] == 0:
#             labels[i] = -100

#     full["labels"] = labels
#     return full

# def tokenize(tokenizer, question: str, answer: str):
#     full_text = f"{question} {answer}{tokenizer.eos_token}"

#     tokenizer.padding_side = "right"
#     tokenizer.pad_token = tokenizer.eos_token
#     full = tokenizer(full_text, padding="max_length", truncation=True, max_length=128)

#     input_ids = full["input_ids"]
#     # Use add_special_tokens=False to avoid extra tokens
#     question_ids = tokenizer(question, add_special_tokens=False)["input_ids"]
#     question_len = len(question_ids)

#     # Create labels: mask out the prompt part
#     labels = [-100] * question_len + input_ids[question_len:]

#     for i in range(len(labels)):
#         if full["attention_mask"][i] == 0:
#             labels[i] = -100

#     full["labels"] = labels
#     return full

# def tokenize(tokenizer, question: str, answer: str):
#     # Add a space between question and answer, as in the full_text
#     full_text = f"{question} {answer}{tokenizer.eos_token}"

#     tokenizer.padding_side = "right"
#     tokenizer.pad_token = tokenizer.eos_token
#     full = tokenizer(full_text, padding="max_length", truncation=True, max_length=128)

#     input_ids = full["input_ids"]
#     # Count tokens in question + space
#     question_prefix = f"{question} "
#     question_len = len(tokenizer(question_prefix, add_special_tokens=False)["input_ids"])

#     # Create labels: mask out the prompt part
#     labels = [-100] * question_len + input_ids[question_len:]

#     for i in range(len(labels)):
#         if full["attention_mask"][i] == 0:
#             labels[i] = -100

#     full["labels"] = labels
#     return full

def tokenize1(tokenizer, question: str, answer: str):
    # Compose the full text as before
    full_text = f"{question} {answer}{tokenizer.eos_token}"

    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    full = tokenizer(full_text, padding="max_length", truncation=True, max_length=128)

    input_ids = full["input_ids"]

    # Tokenize the answer (with no leading space)
    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]

    # Find the start index of answer_ids in input_ids
    def find_sublist(lst, sublst):
        for i in range(len(lst) - len(sublst) + 1):
            if lst[i:i+len(sublst)] == sublst:
                return i
        return -1

    answer_start = find_sublist(input_ids, answer_ids)
    if answer_start == -1:
        # fallback: try with a leading space
        answer_ids = tokenizer(" " + answer, add_special_tokens=False)["input_ids"]
        answer_start = find_sublist(input_ids, answer_ids)
        if answer_start == -1:
            print("Warning: answer tokens not found in input_ids!")
            labels = [-100] * len(input_ids)
        else:
            labels = [-100] * answer_start + input_ids[answer_start:answer_start+len(answer_ids)] + [-100] * (len(input_ids) - answer_start - len(answer_ids))
    else:
        labels = [-100] * answer_start + input_ids[answer_start:answer_start+len(answer_ids)] + [-100] * (len(input_ids) - answer_start - len(answer_ids))

    # Mask out padding
    for i in range(len(labels)):
        if full["attention_mask"][i] == 0:
            labels[i] = -100

    full["labels"] = labels
    return full

def tokenize(tokenizer, question: str, answer: str):
    """
    Tokenize a data element.
    We first append the <EOS> token to the question / answer pair.
    Then we tokenize and construct the ground truth `labels`.
    `labels[i] == -100` for the question or masked out parts, since we only want to supervise
    the answer.
    """
    # print("inside tokenizer")
    # print("question",question)
    # print("ans",answer)
    # print("called")
    full_text = f"{question}{answer}{tokenizer.eos_token}"

    # print("fulltext",full_text)

    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    full = tokenizer(full_text, padding="max_length", truncation=True, max_length=128)

    input_ids = full["input_ids"]
    # print("inputids",input_ids)
    question_len = len(tokenizer(question)["input_ids"])
    # print("qnlen",question_len)

    # Create labels: mask out the prompt part
    labels = [-100] * question_len + input_ids[question_len:]

    # print("labels",labels)

    for i in range(len(labels)):
        if full["attention_mask"][i] == 0:
            labels[i] = -100

    full["labels"] = labels
    return full

