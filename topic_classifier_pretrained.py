# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-9")

possible_topics = ['depression', 'anxiety', 'parenting', 'self-esteem', 'relationship-dissolution', 
                   'workplace-relationships', 'spirituality', 'trauma', 'domestic-violence', 'anger-management', 
                   'sleep-improvement', 'intimacy', 'grief-and-loss', 'substance-abuse', 'family-conflict', 
                   'marriage', 'eating-disorders', 'relationships', 'lgbtq', 'behavioral-change', 
                   'addiction', 'legal-regulatory', 'professional-ethics', 'stress', 
                   'human-sexuality', 'social-relationships', 'children-adolescents',
                   'military-issues', 'self-harm', 'diagnosis', 'counseling-fundamentals']

def get_topics(query):
    return pipe(query, possible_topics)

if __name__ == "__main__":
    # example:
    query = "I'm really feeling hopeless after my breakup..."
    result = pipe(query, possible_topics)

    print(result)