import re
import nltk
import ftfy
import multiprocessing
from datasets import load_dataset
from langdetect import detect_langs
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download("punkt")

whitespace={
    " ",
    " ",
    " ",
    " ",
    " ",
    "　",
    " ",
    " ",
    " ",
    " ",
    "￼",
    "",
}

def is_not_empty(example):
    return len(example['text']) > 0

def is_terminal_punctuation(line):
    return bool(re.search(r'[\.\?!"]\s*$', line))

def is_valid_sentence(sentence):
    words = word_tokenize(sentence)
    return len(words) >= 3

def contains_javascript(sentence):
    return bool(re.search(r'\b(?:java\s*script|JS)\b', sentence, re.IGNORECASE))

def contains_lorem_ipsum(sentence):
    return bool(re.search(r'\b(?:lorem\s*ipsum)\b', sentence, re.IGNORECASE))

def contains_curly_bracket(sentence):
    return bool(re.search(r'[{}]', sentence))

def is_english(sentence):
    languages = detect_langs(sentence)
    return any(lang.lang == 'en' and lang.prob >= 0.99 for lang in languages)

def contains_url(sentence):
    url_pattern = r'(?:(?:http|https|ftp):\/\/|www\.)[\w/\-?=%.]+\.[\w/\-?=%.]+'
    return bool(re.search(url_pattern, sentence))

def contains_phone_number(sentence):
    phone_number_pattern = r'\b(?:\+\d{1,3})?[-. (]*(?:\d{1,3})?[-. )]*(?:\d{2,5})[-. (]*(?:\d{2,5})[-. )]*(?:\d{2,5})\b'
    return bool(re.search(phone_number_pattern, sentence))

def remove_ssn(sentence):
    return re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '', sentence)

def remove_ip_addresses(sentence):
    return re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '', sentence)

def remove_credit_card_numbers(sentence):
    return re.sub(r'\b(?:\d{4}[ -]?){3}\d{4}\b', '', sentence)

def remove_repeated_chars(sentence):
    return re.sub(r'(.)\1{3,}', r'\1', sentence)

def fix_encoding(text):
    return ftfy.fix_text(text)

def normalize_whitespace(text):
    text = "".join([char if char not in whitespace else " " for char in text])
    return re.sub(r'\s+', ' ', text).strip()

def has_min_chars(text, min_chars=3):
    return len(text) >= min_chars

def filter_dataset(example):
    text = example["text"]
    text = fix_encoding(text)

    # Check for lorem ipsum and curly brackets in the initial text
    if contains_lorem_ipsum(text) or contains_curly_bracket(text) or not is_english(text):
        return {"text": ""}

    sentences = sent_tokenize(text)

    valid_sentences = []

    for sentence in sentences:
        if (is_terminal_punctuation(sentence) and
            is_valid_sentence(sentence) and
            not contains_javascript(sentence) and
            not contains_phone_number(sentence) and
            not contains_url(sentence) and
            has_min_chars(sentence)):
            sentence = remove_ssn(sentence)
            sentence = remove_repeated_chars(sentence)
            sentence = remove_ip_addresses(sentence)
            sentence = remove_credit_card_numbers(sentence)
            sentence = normalize_whitespace(sentence)
            valid_sentences.append(sentence)

    if len(valid_sentences) < 5:
        return {"text": ""}
    else:
        return {"text": "\n".join(valid_sentences)}

if __name__ == "__main__":
    dataset = load_dataset("conceptofmind/test_c4_filters", split="train")

    print(dataset)

    filtered_dataset = dataset.map(
        filter_dataset,
        num_proc=multiprocessing.cpu_count()
    ).filter(is_not_empty)

    print(filtered_dataset)
    
    filtered_dataset.push_to_hub('uber_clean')