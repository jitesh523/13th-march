"""Test the notebook's Markov chatbot without starting its interactive loop."""

import ast
import json
from pathlib import Path
import random
import unittest


NOTEBOOK = Path(__file__).resolve().parents[1] / "Untitled-1.ipynb"


def load_chatbot_class():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] == "code" and "class ChatBot:" in source:
            tree = ast.parse(source)
            chatbot = next(node for node in tree.body
                           if isinstance(node, ast.ClassDef) and node.name == "ChatBot")
            namespace = {"random": random}
            exec(compile(ast.Module(body=[chatbot], type_ignores=[]),
                         str(NOTEBOOK), "exec"), namespace)
            return namespace["ChatBot"]
    raise AssertionError("Markov ChatBot class not found in notebook")


class MarkovChainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chatbot_class = load_chatbot_class()

    def make_bot(self, text, n=3):
        # Skip corpus I/O and the welcome screen; exercise the actual notebook methods.
        bot = self.chatbot_class.__new__(self.chatbot_class)
        bot.tokens = text.split()
        bot.n = n
        bot.build_markov_chain()
        return bot

    def test_exactly_one_ngram_can_generate_a_response(self):
        bot = self.make_bot("Mowgli met Bagheera.")
        self.assertEqual(bot.chain, {("Mowgli", "met"): ["Bagheera."]})
        self.assertEqual(bot.generate_response("Mowgli met"), "Mowgli met Bagheera.")
        self.assertEqual(bot.generate_response(), "Mowgli met Bagheera.")

    def test_final_ngram_completes_the_response(self):
        bot = self.make_bot("Mowgli met the wise Bagheera.")
        self.assertEqual(bot.generate_response("Mowgli met"),
                         "Mowgli met the wise Bagheera.")
        self.assertIn(("the", "wise"), bot.keys)

    def test_repeated_final_transition_keeps_its_frequency(self):
        bot = self.make_bot("a b c a b c")
        self.assertEqual(bot.chain[("a", "b")], ["c", "c"])

    def test_bigram_includes_last_word(self):
        bot = self.make_bot("Mowgli met Bagheera.", n=2)
        self.assertEqual(bot.generate_response("Mowgli"), "Mowgli met Bagheera.")

    def test_corpus_shorter_than_order_has_no_transitions(self):
        for text in ("", "Mowgli", "Mowgli met"):
            with self.subTest(text=text):
                bot = self.make_bot(text)
                self.assertEqual(bot.chain, {})
                self.assertEqual(bot.keys, [])


if __name__ == "__main__":
    unittest.main()
