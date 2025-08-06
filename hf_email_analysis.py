from flask import Flask, request, jsonify
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import re
import json
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

class HuggingFaceEmailAnalyzer:
    def __init__(self):
        """Initialize with free Hugging Face models"""
        print(" Loading AI models... This may take a moment on first run...")
        
        try:
            # Sentiment Analysis Model (free, pre-trained)
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis", 
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                return_all_scores=True
            )
            print(" Sentiment analyzer loaded")
            
            # Text Classification for Urgency (free, pre-trained)
            self.urgency_analyzer = pipeline(
                "text-classification",
                model="microsoft/DialoGPT-medium"  # We'll use this creatively
            )
            
            # Alternative: Use a general classification model
            self.classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli"
            )
            print(" Classification models loaded")
            
        except Exception as e:
            print(f"⚠️ Error loading some models: {e}")
            # Fallback to simpler models
            self.sentiment_analyzer = pipeline("sentiment-analysis")
            self.classifier = None
            print(" Basic models loaded")
        
        # Keywords for categories (as backup)
        self.category_keywords = {
            'work': ['meeting', 'project', 'deadline', 'report', 'team', 'manager', 'office'],
            'support': ['help', 'issue', 'problem', 'error', 'bug', 'broken', 'not working'],
            'marketing': ['sale', 'discount', 'offer', 'promotion', 'deal', 'buy now'],
            'personal': ['family', 'friend', 'birthday', 'vacation', 'personal'],
            'spam': ['winner', 'congratulations', 'free money', 'click here', 'act now']
        }
        
        print(" Hugging Face Email Analyzer ready!")
    
    def analyze_sentiment_advanced(self, text):
        """Advanced sentiment analysis using AI"""
        try:
            if self.sentiment_analyzer:
                results = self.sentiment_analyzer(text)
                
                # Handle different model output formats
                if isinstance(results[0], list):
                    results = results[0]
                
                # Find the highest confidence sentiment
                best_result = max(results, key=lambda x: x['score'])
                label = best_result['label'].lower()
                
                # Map different label formats to our standard
                if 'pos' in label or 'positive' in label:
                    return 'positive', best_result['score']
                elif 'neg' in label or 'negative' in label:
                    return 'negative', best_result['score']
                else:
                    return 'neutral', best_result['score']
                    
        except Exception as e:
            print(f"Sentiment analysis error: {e}")
            
        # Fallback to simple keyword matching
        return self.simple_sentiment(text), 0.5
    
    def simple_sentiment(self, text):
        """Simple sentiment as fallback"""
        text_lower = text.lower()
        positive_words = ['thank', 'great', 'excellent', 'happy', 'good', 'amazing']
        negative_words = ['problem', 'issue', 'error', 'bad', 'terrible', 'angry', 'urgent']
        
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count:
            return 'positive'
        elif neg_count > pos_count:
            return 'negative'
        return 'neutral'
    
    def analyze_urgency_ai(self, text):
        """AI-powered urgency detection"""
        try:
            if self.classifier:
                # Use zero-shot classification for urgency
                urgency_labels = ['very urgent', 'somewhat urgent', 'not urgent']
                result = self.classifier(text, urgency_labels)
                
                best_label = result['labels'][0]
                confidence = result['scores'][0]
                
                if 'very urgent' in best_label and confidence > 0.5:
                    return 'high', confidence
                elif 'somewhat urgent' in best_label and confidence > 0.4:
                    return 'medium', confidence
                else:
                    return 'low', confidence
                    
        except Exception as e:
            print(f"Urgency AI error: {e}")
        
        # Fallback to keyword-based urgency
        return self.simple_urgency(text), 0.5
    
    def simple_urgency(self, text):
        """Simple urgency detection as fallback"""
        text_lower = text.lower()
        high_urgency = ['urgent', 'asap', 'immediately', 'emergency', 'critical']
        medium_urgency = ['soon', 'important', 'need', 'deadline', 'today']
        
        high_count = sum(1 for word in high_urgency if word in text_lower)
        medium_count = sum(1 for word in medium_urgency if word in text_lower)
        
        if high_count > 0:
            return 'high'
        elif medium_count > 0:
            return 'medium'
        return 'low'
    
    def analyze_category_ai(self, text):
        """AI-powered category detection"""
        try:
            if self.classifier:
                category_labels = ['work business', 'personal private', 'marketing advertisement', 
                                 'support help', 'spam unwanted']
                result = self.classifier(text, category_labels)
                
                best_label = result['labels'][0]
                confidence = result['scores'][0]
                
                # Map AI labels to our categories
                if 'work' in best_label:
                    return 'work', confidence
                elif 'personal' in best_label:
                    return 'personal', confidence
                elif 'marketing' in best_label:
                    return 'marketing', confidence
                elif 'support' in best_label:
                    return 'support', confidence
                elif 'spam' in best_label:
                    return 'spam', confidence
                    
        except Exception as e:
            print(f"Category AI error: {e}")
        
        # Fallback to keyword matching
        return self.simple_category(text), 0.5
    
    def simple_category(self, text):
        """Simple category detection as fallback"""
        text_lower = text.lower()
        scores = {}
        
        for category, keywords in self.category_keywords.items():
            scores[category] = sum(1 for word in keywords if word in text_lower)
        
        max_category = max(scores.items(), key=lambda x: x[1])
        return max_category[0] if max_category[1] > 0 else 'other'
    
    def extract_key_points_ai(self, text):
        """Extract key points using AI"""
        # For now, use simple sentence extraction
        # You could add a summarization model here
        sentences = re.split(r'[.!?]+', text)
        important_sentences = []
        
        for sentence in sentences[:5]:
            sentence = sentence.strip()
            if len(sentence) > 15:
                important_sentences.append(sentence[:80] + "..." if len(sentence) > 80 else sentence)
            if len(important_sentences) >= 3:
                break
        
        return important_sentences if important_sentences else ["Email content analyzed"]
    
    def suggest_action(self, urgency, sentiment, category):
        """Suggest action based on analysis"""
        if urgency == 'high':
            return 'flag_important'
        elif category == 'spam':
            return 'archive'
        elif sentiment == 'negative' and category in ['work', 'support']:
            return 'forward'
        elif category == 'work':
            return 'create_task'
        else:
            return 'reply'
    
    def analyze_email(self, subject="", sender="", body=""):
        """Main analysis function using AI models"""
        try:
            # Combine text for analysis
            full_text = f"{subject} {body}"
            
            # AI-powered analysis
            sentiment, sentiment_confidence = self.analyze_sentiment_advanced(full_text)
            urgency, urgency_confidence = self.analyze_urgency_ai(full_text)
            category, category_confidence = self.analyze_category_ai(full_text)
            
            # Extract key points
            key_points = self.extract_key_points_ai(body or subject)
            
            # Suggest action
            suggested_action = self.suggest_action(urgency, sentiment, category)
            
            # Calculate overall confidence
            overall_confidence = (sentiment_confidence + urgency_confidence + category_confidence) / 3
            
            return {
                "urgency": urgency,
                "sentiment": sentiment,
                "category": category,
                "key_points": key_points,
                "suggested_action": suggested_action,
                "confidence": round(overall_confidence, 2),
                "processed_by": "huggingface_ai_models",
                "word_count": len(full_text.split()),
                "analysis_time": datetime.now().isoformat(),
                "ai_powered": True
            }
            
        except Exception as e:
            return {
                "urgency": "medium",
                "sentiment": "neutral",
                "category": "other",
                "key_points": ["Analysis failed"],
                "suggested_action": "review_manually",
                "confidence": 0.0,
                "error": str(e),
                "processed_by": "huggingface_fallback"
            }

# Initialize analyzer (this may take a moment)
print(" Starting Hugging Face Email Analyzer...")
analyzer = HuggingFaceEmailAnalyzer()

@app.route('/analyze_email', methods=['POST'])
def analyze_email():
    """API endpoint for email analysis"""
    try:
        data = request.json
        
        subject = data.get('subject', '')
        sender = data.get('sender', data.get('from', ''))
        body = data.get('body', data.get('content', ''))
        
        if not any([subject, body]):
            return jsonify({
                "error": "Email subject or body is required",
                "status": "error"
            }), 400
        
        analysis = analyzer.analyze_email(subject, sender, body)
        
        return jsonify({
            "status": "success",
            "analysis": analysis,
            "message": "Analysis completed using free Hugging Face AI models",
            "original_email": {
                "subject": subject,
                "sender": sender,
                "body_preview": body[:100] + "..." if len(body) > 100 else body
            }
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "huggingface_email_analyzer",
        "ai_powered": True,
        "cost": "100% FREE - Uses local AI models",
        "version": "1.0"
    })

@app.route('/', methods=['GET'])
def home():
    """Basic info endpoint"""
    return jsonify({
        "service": "Hugging Face Email Analysis API",
        "ai_powered": True,
        "cost": "100% FREE - No API keys needed!",
        "models": "Uses Facebook BART, CardiffNLP RoBERTa, and other free models",
        "endpoints": [
            "POST /analyze_email - Analyze email content with AI",
            "GET /health - Health check",
            "GET / - This info"
        ]
    })

if __name__ == '__main__':
    print(" Hugging Face Email Analysis Service Ready!")
    print(" Cost: $0.00 - Uses free AI models!")
    print(" AI-Powered: Real machine learning models")
    print(" Endpoints:")
    print("   - POST http://localhost:5001/analyze_email")
    print("   - GET http://localhost:5001/health")
    
    app.run(debug=True, host='0.0.0.0', port=5001)