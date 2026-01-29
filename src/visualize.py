import matplotlib.pyplot as plt
from load_ct import load_dicom

if __name__ == "__main__":
    path = "data/raw/patient_10/IM-0001-0001.dcm"
    img = load_dicom(path)

    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap="gray")
    plt.title("DICOM Slice: IM-0001-0001.dcm")
    plt.axis("off")
    plt.show()