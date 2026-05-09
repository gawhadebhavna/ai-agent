class SAPIntentClassifier:

    @staticmethod
    def classify(prompt: str):

        prompt_lower = prompt.lower()

        if "metadata" in prompt_lower:
            return "metadata"

        if "schema" in prompt_lower:
            return "metadata"

        if "table structure" in prompt_lower:
            return "metadata"

        if "fetch data" in prompt_lower:
            return "data"

        if "load data" in prompt_lower:
            return "data"

        if "store data" in prompt_lower:
            return "data"

        return "unknown"