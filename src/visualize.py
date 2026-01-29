import matplotlib.pyplot as plt
from load_ct import load_dicom_series

def show_slice(img, title=""):
    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    folder = "data/raw/patient_10"
    series = load_dicom_series(folder)

    print(f"Loaded {len(series)} slices.")
    print("Available indices:", list(range(len(series))))

    # Change this number to scroll through slices
    idx = 2

    path, img = series[idx]
    show_slice(img, title=f"Slice {idx} | {path.split('/')[-1]}")