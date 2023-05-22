import unittest
from datasets import load_dataset
from c4_filters import (
    contains_phone_number,
    contains_javascript,
    contains_lorem_ipsum,
    contains_curly_bracket,
    is_terminal_punctuation,
    is_valid_sentence,
    has_min_chars,
    fix_encoding,
    normalize_whitespace,
    remove_repeated_chars,
    has_min_alphanumeric_percentage,
    is_english,
    remove_ssn,
    remove_credit_card_numbers,
    contains_url,
    remove_ip_addresses,
    filter_dataset
)


class TestFilters(unittest.TestCase):

    def test_contains_lorem_ipsum(self):
        test_cases = [
            ("This is a sentence with lorem ipsum text.", True),
            ("Lorem ipsum dolor sit amet, consectetur adipiscing elit.", True),
            ("This is a normal sentence without it.", False)
        ]

        for sentence, expected_result in test_cases:
            self.assertEqual(contains_lorem_ipsum(sentence), expected_result)

    def test_contains_curly_bracket(self):
        test_cases = [
            ("This is a sentence with a { curly bracket.", True),
            ("The code snippet is: function() { return true; }", True),
            ("This is a normal sentence without any curly brackets.", False)
        ]

        for sentence, expected_result in test_cases:
            self.assertEqual(contains_curly_bracket(sentence), expected_result)

    def test_is_terminal_punctuation(self):
        test_cases = [
            ("This sentence ends with a period.", True),
            ("What a great day!", True),
            ("Is this a question?", True),
            ("This sentence does not have terminal punctuation", False)
        ]

        for sentence, expected_result in test_cases:
            self.assertEqual(is_terminal_punctuation(sentence), expected_result)

    def test_is_valid_sentence(self):
        test_cases = [
            ("This sentence has at least three words.", True),
            ("Only three words.", True),
            ("One.", False)
        ]

        for sentence, expected_result in test_cases:
            self.assertEqual(is_valid_sentence(sentence), expected_result)


    def test_contains_phone_number(self):
        test_cases = [
            ("Call me at (123) 456-7890.", True),
            ("My number is +1 (555) 123-4567.", True),
            ("You can reach me at 9876543210.", True),
            ("The temperature is 100F today.", False),
            ("This is a regular sentence without a phone number.", False),
            ("The number you are trying to reach is no longer in service.", False)
        ]

        for sentence, expected_result in test_cases:
            self.assertEqual(contains_phone_number(sentence), expected_result)


    def test_contains_javascript(self):
        test_cases = [
            ("This page requires JavaScript to run properly.", True),
            ("Please enable Javascript in your browser.", True),
            ("This is a sentence with the word javascript.", True),
            ("This page uses CSS for styling.", False),
            ("This is a regular sentence without the word.", False),
            ("Java and Python are popular programming languages.", False)
        ]

        for sentence, expected_result in test_cases:
            self.assertEqual(contains_javascript(sentence), expected_result)


    def test_has_min_chars(self):
        test_cases = [
            ("This sentence has more than 20 characters.", True),
            ("This sentence has less.", True)
        ]

        for sentence, expected_result in test_cases:
            self.assertEqual(has_min_chars(sentence, 20), expected_result)

    def test_fix_encoding(self):
        test_cases = [
            ("This is a normal sentence without any encoding issues.", "This is a normal sentence without any encoding issues."),
            ("This sentence has â€˜smartâ€™ quotes.", "This sentence has 'smart' quotes."),
            ("MÃ¶bius strip is a surface with only one side.", "Möbius strip is a surface with only one side.")
        ]

        for input_text, expected_result in test_cases:
            self.assertEqual(fix_encoding(input_text), expected_result)

    def test_normalize_whitespace(self):
        test_cases = [
            ("This is a normal sentence without any unusual whitespace.", "This is a normal sentence without any unusual whitespace."),
            ("This sentence has\u2009different\u200aspaces.", "This sentence has different spaces."),
            ("This\u3000sentence\u2002has\u2003wide\u2004spaces.", "This sentence has wide spaces.")
        ]

        for input_text, expected_result in test_cases:
            self.assertEqual(normalize_whitespace(input_text), expected_result)

    def test_remove_repeated_chars(self):
        test_cases = [
            ("This is a normal sentence without any repeated characters.", "This is a normal sentence without any repeated characters."),
            ("Thiiiiis sentenceeee has soooome repeaaaated chaaaaracters.", "This sentence has some repeated characters."),
            ("AAAAAhhhhhh, I can't believe thisssss!", "Ah, I can't believe this!")
        ]

        for input_text, expected_result in test_cases:
            self.assertEqual(remove_repeated_chars(input_text), expected_result)

    def test_has_min_alpha_numeric(self):
        test_cases = [
            ("This is a normal sentence with enough alpha numeric characters.", True),
            ("$%#@!&*^", False),
            ("A sentence with 20% alpha numeric characters.", True),
            ("A12_+%# $()?", False)
        ]

        for input_text, expected_result in test_cases:
            self.assertEqual(has_min_alphanumeric_percentage(input_text, min_percentage=75), expected_result)

    def test_is_english(self):
        test_cases = [
            ("This is a normal English sentence.", True),
            ("Ceci est une phrase en français.", False),
            ("Dies ist ein Satz auf Deutsch.", False),
            ("Esta es una oración en español.", False)
        ]

        for input_text, expected_result in test_cases:
            self.assertEqual(is_english(input_text), expected_result)

    def test_remove_ssn(self):
        test_cases = [
            ("This is a normal sentence without any social security numbers.", "This is a normal sentence without any social security numbers."),
            ("My social security number is 123-45-6789.", "My social security number is ."),
            ("Another SSN is 987-65-4321, please handle it.", "Another SSN is , please handle it.")
        ]

        for input_text, expected_result in test_cases:
            self.assertEqual(remove_ssn(input_text), expected_result)

    def test_contains_url(self):
        test_cases = [
            ("Visit our website at https://www.example.com.", True),
            ("You can find the article at http://example.org/article.", True),
            ("Check out our blog: www.blog.example.net", True),
            ("My email is john@example.com", False),
            ("The price is $20,000.", False),
            ("This is a regular sentence without a URL.", False)
        ]

        for sentence, expected_result in test_cases:
            self.assertEqual(contains_url(sentence), expected_result)

    def test_remove_credit_card(self):
        test_cases = [
            ("This is a normal sentence without any credit card numbers.", "This is a normal sentence without any credit card numbers."),
            ("My credit card number is 1234-5678-9123-4567.", "My credit card number is ."),
            ("Another credit card number is 9876-5432-1098-7654, please remove it.", "Another credit card number is , please remove it.")
        ]

        for input_text, expected_result in test_cases:
            self.assertEqual(remove_credit_card_numbers(input_text), expected_result)

    def test_remove_ip(self):
        test_cases = [
            ("This is a normal sentence without any IP addresses.", "This is a normal sentence without any IP addresses."),
            ("The server IP address is 192.168.1.1.", "The server IP address is ."),
            ("Another IP address is 10.0.0.1, please remove it.", "Another IP address is , please remove it.")
        ]

        for input_text, expected_result in test_cases:
            self.assertEqual(remove_ip_addresses(input_text), expected_result)

    def test_filter_dataset(self):
        # Load a sample from the dataset
        dataset = load_dataset("conceptofmind/test_l", split="train")
        example = dataset[0]  # Get the first example from the dataset

        # Apply the filter_dataset function
        filtered_example = filter_dataset(example)

        # Perform assertions to check if the filtered_example is as expected
        # Example: Check if the filtered_example is not empty
        self.assertNotEqual(filtered_example["text"], "")



if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)