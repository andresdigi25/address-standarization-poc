import nltk
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.chunk import tree2conlltags

# Download required NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

def extract_address_nltk(address: str) -> dict:
    # Initialize components
    components = {
        "street": None,
        "number": None,
        "city": None,
        "state": None,
        "zip_code": None
    }
    
    # Tokenize and tag the text
    tokens = word_tokenize(address)
    tagged = pos_tag(tokens)
    named_entities = ne_chunk(tagged)
    
    # Convert tree to list of tuples
    iob_tagged = tree2conlltags(named_entities)
    
    current_entity = []
    current_label = None
    
    # Process each token
    for word, pos, ne in iob_tagged:
        if word.isdigit() and len(word) == 5:
            components["zip_code"] = word
        elif word.isdigit() and not components["number"]:
            components["street"] = None  # Reset street if we find a number
            components["number"] = word
        elif ne.startswith('B-GPE'):
            if not components["city"]:
                components["city"] = word
            elif not components["state"] and len(word) == 2:
                components["state"] = word
        elif ne.startswith('I-GPE'):
            if components["city"]:
                components["city"] += f" {word}"
        elif not components["street"] and pos.startswith('NN'):
            if components["number"]:
                components["street"] = word
            
    return components
