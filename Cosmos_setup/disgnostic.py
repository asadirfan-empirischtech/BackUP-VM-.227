import requests
import json

try:
    res = requests.get("http://localhost:8000/openapi.json")
    openapi = res.json()

    print("\n🔍 EXAMINING /v1/videos/sync ENDPOINT...")
    content = openapi['paths']['/v1/videos/sync']['post']['requestBody']['content']
    
    if 'application/json' in content:
        print("-> Server expects JSON.")
        ref = content['application/json']['schema'].get('$ref')
        if ref:
            model = ref.split('/')[-1]
            print(f"-> Schema for {model}:")
            print(json.dumps(openapi['components']['schemas'][model], indent=2))
        else:
            print(json.dumps(content['application/json']['schema'], indent=2))
            
    elif 'multipart/form-data' in content or 'application/x-www-form-urlencoded' in content:
        print(f"-> Server expects FORM DATA ({list(content.keys())}), not JSON.")
    else:
        print(f"-> Server expects: {list(content.keys())}")

except Exception as e:
    print(f"Error fetching schema: {e}")
