def preprocess_qwen(sources, tokenizer, has_image: bool = False, max_len=2048, system_message: str = "You are a helpful assistant.") -> dict:
    IMAGE_TOKEN_INDEX = -300
    NEWLINE_TOKEN = 198
    
    roles = {"human": "user", "gpt": "assistant"}
    processed_ids = []
    
    for source in sources:
        # Combine messages into single string
        input_id = ""
        for i, message in enumerate(source):
            if message['from'] != 'human' and i == 0:
                continue
            input_id += roles[message['from']] + " " + message['value'] + "\n"
            
        # Remove the last newline
        if input_id.endswith('\n'):
            input_id = input_id[:-1]
            
        # Split the text into parts, separating image tags
        parts = []
        current_pos = 0
        
        while True:
            img_start = input_id.find("<image>", current_pos)
            if img_start == -1:
                # Add remaining text if any
                if current_pos < len(input_id):
                    parts.append(("text", input_id[current_pos:]))
                break
                
            # Add text before image tag if any
            if img_start > current_pos:
                parts.append(("text", input_id[current_pos:img_start]))
                
            # Add image token
            parts.append(("image", None))
            
            # Move position to after image tag
            current_pos = img_start + len("<image>")
        
        # Process each part and build final token list
        final_tokens = []
        
        for part_type, part_text in parts:
            if part_type == "image":
                final_tokens.append(IMAGE_TOKEN_INDEX)
            else:
                # Replace \n with NEWLINE_TOKEN
                text_parts = part_text.split('\n')
                for i, text_part in enumerate(text_parts):
                    if text_part:  # Only process non-empty text
                        tokens = tokenizer(text_part)  # Assume this returns a list of integers
                        final_tokens.extend(tokens)
                    if i < len(text_parts) - 1:  # Add newline token between parts, but not at the end
                        final_tokens.append(NEWLINE_TOKEN)
        
        processed_ids.append(final_tokens)
    
    return processed_ids

class MockTokenizer:
    def __init__(self):
        # Pre-define some common tokens with fixed IDs
        self.token_map = {
            "user": 1000,
            "assistant": 1001,
            "What's": 1002,
            "in": 1003,
            "this": 1004,
            "image": 1005,
            "I": 1006,
            "see": 1007,
            "a": 1008,
            "dog": 1009,
            "playing": 1010,
            "the": 1011,
            "park": 1012,
            ".": 1013,
            "?": 1014,
            " ": 1015,
        }
        self.next_token_id = 2000

    def __call__(self, text):
        tokens = []
        # Split by spaces but keep punctuation
        words = []
        current_word = ""
        
        for char in text:
            if char.isspace():
                if current_word:
                    words.append(current_word)
                    current_word = ""
                words.append(" ")
            elif char in ".,!?":
                if current_word:
                    words.append(current_word)
                    current_word = ""
                words.append(char)
            else:
                current_word += char
        if current_word:
            words.append(current_word)

        # Convert words to tokens
        for word in words:
            if word not in self.token_map:
                self.token_map[word] = self.next_token_id
                self.next_token_id += 1
            tokens.append(self.token_map[word])
            
        return tokens

# Test the implementation
if __name__ == "__main__":
    sources = [
        [
            {"from": "human", "value": "<image> What's in this image?"}, 
            {"from": "gpt", "value": "I see a dog playing in the park."}
        ],
        [
            {"from": "human", "value": "<image> What's in this image?"}, 
            {"from": "gpt", "value": "I see a dog playing in the park."}
        ]    
    ]

    tokenizer = MockTokenizer()
    
    # First, let's test the tokenizer alone
    print("\nTesting tokenizer alone:")
    test_text = "What's in this image?"
    tokens = tokenizer(test_text)
    print(f"Text: {test_text}")
    print(f"Tokens: {tokens}")
    
    # Now test the full preprocessing
    print("\nTesting full preprocessing:")
    
    processed = preprocess_qwen(sources, tokenizer)
    
    print("\nProcessed output:")
    for i, tokens in enumerate(processed):
        print(f"\nSource {i + 1} tokens:")
        print(tokens)
        
        # Print token meanings for better understanding
        print("\nToken meanings:")
        special_tokens = {-300: "<image>", 198: "\\n"}
        token_to_word = {v: k for k, v in tokenizer.token_map.items()}
        
        for token in tokens:
            if token in special_tokens:
                print(f"{token}: {special_tokens[token]}")
            else:
                print(f"{token}: {token_to_word.get(token, 'Unknown')}")