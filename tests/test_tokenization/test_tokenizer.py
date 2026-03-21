from src.tokenization.tokenizer import Tokenizer


def test_character_tokenizer_special_tokens():
    tokenizer = Tokenizer("char", corpus_text="hello world")
    assert tokenizer.pad_token_id == 0
    assert tokenizer.mask_token_id == 1
    encoded = tokenizer.encode("hello")
    assert tokenizer.decode(encoded) == "hello"
