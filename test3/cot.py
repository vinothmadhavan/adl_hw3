from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    # def format_prompt(self, question: str) -> str:
    #     """
    #     Take a question and convert it into a chat template. The LLM will likely answer much
    #     better if you provide a chat template. self.tokenizer.apply_chat_template can help here
    #     """

    #     raise NotImplementedError()
    
    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into a chat template. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        """
        # Example: "How many grams are there per 2 kg?" -> "1 kg = 1000 grams. 2 * 1000 = <answer>2000</answer>"
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that answers unit conversion questions. Be concise and show your reasoning step by step. Wrap the final answer in <answer></answer> tags."
            },
            {
                "role": "user",
                "content": "How many grams are there per 2 kg?"
            },
            {
                "role": "assistant",
                "content": "1 kg = 1000 grams. 2 * 1000 = <answer>2000</answer>"
            },
            {
                "role": "user",
                "content": question
            }
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        return prompt


def load() -> CoTModel:
    return CoTModel()


def test_model():
    from .data import Dataset, benchmark

    testset = Dataset("valid")
    model = CoTModel()
    benchmark_result = benchmark(model, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model, "load": load})
