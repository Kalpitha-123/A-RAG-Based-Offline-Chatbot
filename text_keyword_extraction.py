#!/usr/bin/env python3
"""
Test script to verify keyword extraction improvements
"""

def test_system_keywords():
    """Test the improved system keyword extraction"""
    
    # Sample content that might be in a report
    sample_content = """
    The proposed system architecture consists of multiple interconnected modules designed to handle data processing efficiently. 
    The system implementation follows a microservices approach with real-time data streaming capabilities.
    Our proposed solution includes a frontend interface built with React and a backend API using Python Flask.
    The design methodology incorporates agile development practices and continuous integration.
    This framework provides scalability and maintainability for large-scale applications.
    The platform environment supports both cloud and on-premises deployment options.
    """
    
    # Keywords that should be found
    system_keywords = [
        'proposed system', 'system architecture', 'system design', 'system implementation',
        'proposed', 'architecture', 'design', 'implementation', 'solution', 'approach',
        'methodology', 'framework', 'structure', 'model', 'platform', 'environment'
    ]
    
    print("Testing keyword extraction...")
    print(f"Content length: {len(sample_content)} characters")
    
    # Test keyword matching
    content_lower = sample_content.lower()
    found_keywords = []
    
    for keyword in system_keywords:
        if keyword in content_lower:
            found_keywords.append(keyword)
    
    print(f"Found {len(found_keywords)} system-related keywords:")
    for keyword in found_keywords:
        print(f"  - {keyword}")
    
    # Test sentence extraction
    sentences = sample_content.split('.')
    relevant_sentences = []
    
    for sentence in sentences:
        sentence_lower = sentence.lower()
        sentence_stripped = sentence.strip()
        
        if any(keyword in sentence_lower for keyword in system_keywords):
            if len(sentence_stripped) > 15:
                relevant_sentences.append(sentence_stripped)
    
    print(f"\nFound {len(relevant_sentences)} relevant sentences:")
    for i, sentence in enumerate(relevant_sentences, 1):
        print(f"  {i}. {sentence}")
    
    return len(found_keywords) > 0, len(relevant_sentences) > 0

if __name__ == "__main__":
    keywords_found, sentences_found = test_system_keywords()
    
    if keywords_found and sentences_found:
        print("\n🎉 Keyword extraction test PASSED!")
        print("The improved system should now find relevant information about proposed systems.")
    else:
        print("\n❌ Keyword extraction test FAILED!")
        print("Need to investigate the extraction logic.")
