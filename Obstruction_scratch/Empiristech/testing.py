import os
import requests

# List of image URLs
urls = [
    "https://carfromjapan.com/wp-content/uploads/2018/01/image-1-8-1024x683.jpg",
    "https://www.shutterstock.com/image-photo/damaged-road-cracked-asphalt-sword-260nw-1134666281.jpg",
    "https://carfromjapan.com/wp-content/uploads/2018/01/image2-1024x717.jpg",
    "https://media.istockphoto.com/id/465926255/photo/damaged-road.jpg?s=612x612&w=0&k=20&c=BpAIGaTwkmxrlJEJlpKIWtd1ccKITuozvaRxXMj3Zr0=",
    "https://media.istockphoto.com/id/2160160654/photo/bad-road-cracked-asphalt-with-potholes-and-big-holes-potholes-on-the-road-with-stones-on-the.jpg?s=612x612&w=0&k=20&c=giQZNILtfIi0u2_hCtJIpeMChibO1c4fPJbNDVlEYh0=",
    "https://i.dawn.com/primary/2022/06/62b0ecfc28a55.jpg",
    "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQXqd-plPloirh4YtgweDW8BNqIBrlEoPku57e7VvxZPgVBIt3KTgJbRKHA&s=10",
    "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSs_eJ24XNfaUWcPVfQmNeYqh3Sime3Mg1Auxt7zUY9U1M4q42Wmw8g3kh7&s=10",
    "https://i.dawn.com/large/2021/03/6057a843991c0.jpg",
    "https://media.istockphoto.com/id/2093257835/photo/mountain-landslide.jpg?s=612x612&w=0&k=20&c=rwJXGYaZfW3iT4DOhpRG2tkzQ6kJHkjA9GGWurUIiGs=",
    "https://media.istockphoto.com/id/2152899790/photo/road-closure-and-signposts.jpg?s=612x612&w=0&k=20&c=uBHPzPKaA29-gbVeHZfHfxTs7coOLvsnLDwz-O6Gv-A=",
    "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSIz8n8jMEH0lXKFVy419mPWx7LCC9e6UftdqWqn7fO1JyI7Y5v83hhUn95&s=10"
]

# Set a User-Agent header to prevent getting blocked by websites/CDNs
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# Create a folder to store downloaded images (optional)
output_dir = "downloaded_images"
os.makedirs(output_dir, exist_ok=True)

for i, url in enumerate(urls, start=1):
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise error for bad HTTP responses
        
        # Define file path
        filename = os.path.join(output_dir, f"image_{i}.jpg")
        
        # Write the content to a file in binary mode
        with open(filename, "wb") as f:
            f.write(response.content)
            
        print(f"Successfully downloaded: image_{i}.jpg")
        
    except Exception as e:
        print(f"Failed to download image {i} from {url}. Error: {e}")

print("\nDownload process completed!")