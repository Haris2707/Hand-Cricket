from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Optional

class CodeGenerator:
    def __init__(self):
        """कोड जनरेटर को इनिशियलाइज करें"""
        # CodeLlama मॉडल लोड करें - यह कोड जनरेशन के लिए विशेष रूप से ट्रेंड है
        self.model_name = "codellama/CodeLlama-7b-hf"
        print("मॉडल लोड हो रहा है...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        print("मॉडल लोड हो गया!")
        
    def generate_code(self, prompt: str, language: Optional[str] = None) -> str:
        """यूजर के प्रॉम्प्ट से कोड जनरेट करें"""
        try:
            # प्रॉम्प्ट को फॉर्मेट करें
            formatted_prompt = self._format_prompt(prompt, language)
            
            # टोकनाइज करें और जनरेट करें
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)
            
            # कोड जनरेट करें
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=1000,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # आउटपुट को डीकोड करें
            generated_code = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return generated_code.replace(formatted_prompt, "").strip()
            
        except Exception as e:
            return f"एरर: {str(e)}"
    
    def _format_prompt(self, prompt: str, language: Optional[str]) -> str:
        """प्रॉम्प्ट को फॉर्मेट करें"""
        if language:
            return f"Write code in {language} for the following task:\n{prompt}\n\nCode:\n"
        return f"{prompt}\n\nCode:\n"

def main():
    # जनरेटर का इंस्टेंस बनाएं
    generator = CodeGenerator()
    
    print("कोड जनरेटर में आपका स्वागत है!")
    print("कोड जनरेट करने के लिए अपना प्रॉम्प्ट लिखें ('quit' से बाहर निकलें)")
    
    while True:
        prompt = input("\nआपका प्रॉम्प्ट: ")
        if prompt.lower() == 'quit':
            break
            
        language = input("प्रोग्रामिंग भाषा (वैकल्पिक): ").strip() or None
        
        print("\nकोड जनरेट हो रहा है...")
        generated_code = generator.generate_code(prompt, language)
        print("\nजनरेटेड कोड:")
        print("-" * 50)
        print(generated_code)
        print("-" * 50)

if __name__ == "__main__":
    main() 