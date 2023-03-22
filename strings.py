TITLE = "Alpaca-LoRA Playground"

ABSTRACT = """
This is for internal Use only. By ORI employees.
"""

BOTTOM_LINE = """
In order to process batch generation, the common parameters in LLaMA are fixed. If you want to change the values of them, please do that in `generation_config.yaml`
"""

DEFAULT_EXAMPLES = [
    {
        "title": "1️⃣ List all Indian provinces in alphabetical order.",
        "examples": [
            ["1", "List all Indian provinces in alphabetical order."],
            ["2", "Which ones are on the east side?"],
            ["3", "What foods are famous in each province?"],
            ["4", "What about sightseeing? or landmarks?"],
        ],
    },
    {
        "title": "2️⃣ Tell me about Alpacas.",
        "examples": [
            ["1", "Tell me about alpacas."],
            ["2", "What other animals are living in the same area?"],
            ["3", "Are they the same species?"],
            ["4", "Write a Python program to return those species"],
        ],
    },
    {
        "title": "3️⃣ Tell me about the One-piece.",
        "examples": [
            ["1", "Tell me about Luffy."],
        ]
    },
    {
        "title": "4️⃣ Write a Python program that prints the first 10 Fibonacci numbers.",
        "examples": [
            ["1", "Write a Python program that prints the first 10 Fibonacci numbers."],
            ["2", "could you explain how the code works?"]            
        ]
    }
]

SPECIAL_STRS = {
    "continue": "continue.",
    "summarize": "summarize our conversations so far in three sentences."
}