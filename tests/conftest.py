import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Tiny hand-written toy reviews, ONLY for testing code paths. They say nothing about
# real-world accuracy; real results come from the BanglaBook dataset (see README).
POSITIVE = [
    "বইটি খুব ভালো লেগেছে", "অসাধারণ একটি বই", "চমৎকার লেখা, দারুণ লাগলো",
    "সেবার মান খুবই ভালো পেয়েছি", "অনেক সুন্দর বই, সবাইকে পড়তে বলবো",
    "দারুণ প্রিন্ট এবং চমৎকার কাগজ", "খুব উপকারী বই, ধন্যবাদ",
    "এই লেখকের সেরা বই, অসাধারণ",
]
NEUTRAL = [
    "বইটি মোটামুটি", "সাধারণ একটি বই", "ঠিকঠাক আছে, তেমন কিছু না",
    "মাঝামাঝি মানের বই", "খারাপও না ভালোও না, মোটামুটি",
    "একবার পড়া যায়, সাধারণ মান", "গড়পড়তা লেখা",
    "দাম অনুযায়ী মোটামুটি",
]
NEGATIVE = [
    "বইটি একদম বাজে", "খুব খারাপ প্রিন্ট, টাকা নষ্ট", "জঘন্য মানের বই, কিনে ঠকেছি",
    "পাতা ছেঁড়া এবং বাজে কাগজ", "একদম পড়ার যোগ্য না, বিরক্তিকর",
    "অত্যন্ত নিম্নমানের বই", "হতাশ হয়েছি, খারাপ লেখা",
    "প্রতারণা, ভুলেও কিনবেন না",
]


@pytest.fixture
def toy_data():
    texts = NEGATIVE + NEUTRAL + POSITIVE
    labels = [0] * len(NEGATIVE) + [1] * len(NEUTRAL) + [2] * len(POSITIVE)
    return texts, labels


@pytest.fixture
def toy_csv_dir(tmp_path, toy_data):
    texts, labels = toy_data
    df = pd.DataFrame({"Review": texts, "label": labels, "Rating": 3})
    for split in ("train", "validation", "test"):
        df.to_csv(tmp_path / f"{split}.csv", index=False)
    return tmp_path
