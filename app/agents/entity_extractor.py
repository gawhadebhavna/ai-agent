class EntityExtractor:

    @staticmethod
    def extract(prompt: str):

        prompt_lower = prompt.lower()

        if "customer" in prompt_lower:
            return "customer"

        if "business partner" in prompt_lower:
            return "customer"

        if "product" in prompt_lower:
            return "product"

        if "sales order" in prompt_lower:
            return "sales_order"

        return None