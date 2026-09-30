import os
import requests

def download_images():
    urls = [
        "https://miro.medium.com/1*GeOMS7vkZsp0zNKYtdodDQ.jpeg",
        "https://cdn.pixabay.com/photo/2022/04/18/12/16/highway-7140352_1280.jpg",
        "https://media.istockphoto.com/id/157481087/photo/fallen-tree-blocking-road.jpg?s=612x612&w=0&k=20&c=PxqjhdPyNVtCIOhA8U-wWKH42wMvJvnuWzhRRXML4Fw=",
        "https://storage.googleapis.com/kagglesdsdata/datasets/1375820/12527284/Indian_vehicle_dataset/20210521_09_44_00_000_OgfflYWVtcZg5UlVKsTtIh0krh83_F_4016_3008.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20260812%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20260812T125948Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=050bb3bb332e11f068c18c988996127f75df5600096760d637f2fe9bbc19bdcb6242105d1ff0175180c117f6b4dc98cf7399c48698023601b7c31e77da6efe2e933c0a19fe548494c9d03723b095d8f7769c3deaddf86838552f48084f4a267b848d0f536b595cdc8a957e3a126258d439e3b17487392fc851c3fa3cace5fd4b74db326e4eee2559b721012123ca5a7480fb64cb98cd8a6b5ef34642ab7ed4354d580c84121ae6b7c1ad8d5e7e3c6cd5aefa357b2df4c7b933c3a76a853bbc582b54edf429831238415435a14cac6f77ed2f8a8dcf7812f4d38ce627c0b77c3dbf7cfb4dc601ddce2edd48a137e4e638a47fdfc883bfd78d577ce7570c640ac1",
        "https://storage.googleapis.com/kaggle-datasets-images/8079704/12780041/c56be4d75626dce1e34c3742f55d6fa5/dataset-cover.png?t=2025-08-16-20-39-58",
        "https://storage.googleapis.com/kaggle-datasets-images/8751663/13753570/ea600927c8d24b8b5a502a6ce403fb44/dataset-thumbnail.jpg?t=2025-11-16-12-01-19",
        "https://encrypted-tbn3.gstatic.com/licensed-image?q=tbn:ANd9GcTh_yz1clDXdbt1TJ2W7Oe2u9MjJGloKPGHUiNC3idzDhelI2S9RBq_mIoWNZGBa3kjR8ML7Zdw7-nAie4",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSYfTqlkhFa2ARk6OceOd_GCG8qi2qqteYIlsV3q2lJwcorHJTJodELQvc&s=10",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSfX78WCE5vFL7kaG7E3EEU2HdHNHpDQWdcozqMYJVHFkqfd79VvpQVxrs&s=10",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ5NESyPc6oI_XWTBByM6JxTkDgy9JI55eUp0Y72u6jBpM0dbi6ZQgJdWYM&s=10",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQMyF-09bqVo2wCCg_m7h4sC6xP1CJk57jyP9xNTkbeKqflVY4gSx6VlKZw&s=10",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQMyF-09bqVo2wCCg_m7h4sC6xP1CJk57jyP9xNTkbeKqflVY4gSx6VlKZw&s=10",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS37YgimSplQ8tRRsnRaHYCm1TlMewxGLSjxW4_t59N-K4QpD204-xT4b0&s=10",
        "https://www.shutterstock.com/image-photo/galle-fort-sri-lanka-february-260nw-2747681671.jpg",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT84dlNJ5-UT297U3Y7Otjv0v6g3wLTgGqKSWHvQTYdd5tlbfQZRm74ihQ&s=10",
        "https://media.istockphoto.com/id/522315433/photo/cars-entering-in-tunnel.jpg?s=170667a&w=0&k=20&c=_rwLqCcjKu3ytBI_86bXqqpF6tHe9hWXStqRdeJk_PQ=",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTzPI-mzn9coqISY0xnW9DMcM6hqANiO0eQhiYFn8NMdemCw7ZSN_QJ-d8&s=10",
        "https://media.springernature.com/m685/springer-static/image/art%3A10.1007%2Fs41064-023-00260-0/MediaObjects/41064_2023_260_Fig1_HTML.jpg",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTfN3Yb52JRQGZuHpJQ1G9Zc6c_pO9uSyCfNQVFSXYIo5OmSq4kSB5mVgtN&s=10"
    ]

    # Remove exact duplicate URLs
    unique_urls = list(dict.fromkeys(urls))

    # Define where the images will be saved
    destination_dir = "/home/azureuser/Documents/Research Model testing/Research Paper No 2/dir 3"
    os.makedirs(destination_dir, exist_ok=True)
    
    print(f"Starting download of {len(unique_urls)} images to: {destination_dir}\n")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    for i, url in enumerate(unique_urls):
        try:
            # Adding a User-Agent header helps bypass basic scraping protections on sites like Shutterstock/iStock
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status() 
            
            # Determine extension (default to jpg)
            ext = ".jpg"
            if ".png" in url.lower():
                ext = ".png"
            elif ".jpeg" in url.lower():
                ext = ".jpeg"
            
            # Format filename to keep the directory clean (e.g., image_01.jpg, image_02.png)
            filename = f"image_{i+1:02d}{ext}"
            file_path = os.path.join(destination_dir, filename)
            
            with open(file_path, 'wb') as f:
                f.write(response.content)
                
            print(f"[{i+1}/{len(unique_urls)}] Successfully saved: {filename}")
            
        except requests.exceptions.RequestException as e:
            print(f"[{i+1}/{len(unique_urls)}] Failed to download. Error: {e}")

    print("\nDownload operation complete.")

if __name__ == "__main__":
    download_images()