import os
import shutil
import random

def setup_split():
    # Set seed for reproducibility
    random.seed(42)
    
    # Paths
    kvasir_img_dir = 'dataset/Kvasir-SEG/images/'
    kvasir_mask_dir = 'dataset/Kvasir-SEG/masks/'
    clinic_img_dir = 'dataset/CVC-ClinicDB/images/'
    clinic_mask_dir = 'dataset/CVC-ClinicDB/masks/'
    
    output_dir = 'dataset/PraNet_Split'
    
    # Train directories
    train_img_dir = os.path.join(output_dir, 'TrainDataset', 'images')
    train_mask_dir = os.path.join(output_dir, 'TrainDataset', 'masks')
    
    # Test directories
    test_kvasir_img = os.path.join(output_dir, 'TestDataset', 'Kvasir', 'images')
    test_kvasir_mask = os.path.join(output_dir, 'TestDataset', 'Kvasir', 'masks')
    test_clinic_img = os.path.join(output_dir, 'TestDataset', 'ClinicDB', 'images')
    test_clinic_mask = os.path.join(output_dir, 'TestDataset', 'ClinicDB', 'masks')
    
    # Create all directories
    for d in [train_img_dir, train_mask_dir, test_kvasir_img, test_kvasir_mask, test_clinic_img, test_clinic_mask]:
        os.makedirs(d, exist_ok=True)
        
    def get_files(img_dir):
        return sorted([f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

    kvasir_files = get_files(kvasir_img_dir)
    clinic_files = get_files(clinic_img_dir)
    
    print(f"Found {len(kvasir_files)} Kvasir images and {len(clinic_files)} ClinicDB images.")
    
    # Shuffle files
    random.shuffle(kvasir_files)
    random.shuffle(clinic_files)
    
    # PraNet Split Configuration
    # Kvasir: 900 Train, 100 Test
    # ClinicDB: 550 Train, 62 Test
    kvasir_train = kvasir_files[:900]
    kvasir_test = kvasir_files[900:]
    
    clinic_train = clinic_files[:550]
    clinic_test = clinic_files[550:]
    
    print("Copying Kvasir Train...")
    for f in kvasir_train:
        # Prefix with kvasir_ to avoid naming collisions with ClinicDB
        new_name = f"kvasir_{f}"
        shutil.copy(os.path.join(kvasir_img_dir, f), os.path.join(train_img_dir, new_name))
        shutil.copy(os.path.join(kvasir_mask_dir, f), os.path.join(train_mask_dir, new_name))
        
    print("Copying ClinicDB Train...")
    for f in clinic_train:
        new_name = f"clinic_{f}"
        shutil.copy(os.path.join(clinic_img_dir, f), os.path.join(train_img_dir, new_name))
        shutil.copy(os.path.join(clinic_mask_dir, f), os.path.join(train_mask_dir, new_name))
        
    print("Copying Kvasir Test...")
    for f in kvasir_test:
        shutil.copy(os.path.join(kvasir_img_dir, f), os.path.join(test_kvasir_img, f))
        shutil.copy(os.path.join(kvasir_mask_dir, f), os.path.join(test_kvasir_mask, f))
        
    print("Copying ClinicDB Test...")
    for f in clinic_test:
        shutil.copy(os.path.join(clinic_img_dir, f), os.path.join(test_clinic_img, f))
        shutil.copy(os.path.join(clinic_mask_dir, f), os.path.join(test_clinic_mask, f))

    print(f"Setup Complete!")
    print(f"Train Dataset: {len(os.listdir(train_img_dir))} images (900 Kvasir + 550 ClinicDB)")
    print(f"Test Kvasir: {len(os.listdir(test_kvasir_img))} images")
    print(f"Test ClinicDB: {len(os.listdir(test_clinic_img))} images")

if __name__ == "__main__":
    setup_split()
