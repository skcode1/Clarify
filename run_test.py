import os
from clarify.core import enhance

def test_run():
    # 1. Place a real image named 'input.jpg' or 'input.png' in this directory
    input_image = "input.jpeg"  
    
    if not os.path.exists(input_image):
        print(f"❌ Error: Please drop a real image named '{input_image}' into this folder first!")
        return

    print("🚀 Running Clarify AI Enhancement (EDSR Model)...")
    try:
        # 2. Call the function directly
        # Mode options: "ai" (Super-Resolution) or "classical" (OpenCV Sharpening)
        output_path = enhance(input_image, mode="ai", scale=2)
        
        print(f"✅ Success! Enhanced image saved to: {output_path}")
        
    except Exception as e:
        print(f"❌ An error occurred during enhancement: {e}")

if __name__ == "__main__":
    test_run()
