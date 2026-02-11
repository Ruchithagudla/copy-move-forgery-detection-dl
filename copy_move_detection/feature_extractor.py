import cv2
import numpy as np
from skimage.util import view_as_blocks
from skimage.feature import local_binary_pattern
from scipy.stats import skew, kurtosis

class FeatureExtractor:
    def __init__(self, block_size=32):
        self.block_size = block_size
        
    def extract(self, image_path):
        """Main feature extraction method"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
            
        features = []
        
        # 1. Color Channel Statistics
        features.extend(self._color_features(img))
        
        # 2. Texture Analysis
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features.extend(self._texture_features(gray))
        
        # 3. Block-Based Features (for copy-move detection)
        features.extend(self._block_features(gray))
        
        # 4. Edge and Forgery Artifacts
        features.extend(self._tampering_features(img))
        
        return np.array(features)
    
    def _color_features(self, img):
        """Color distribution features"""
        features = []
        for channel in range(3):  # BGR channels
            hist = cv2.calcHist([img], [channel], None, [256], [0, 256])
            hist = hist / hist.sum()  # Normalize
            
            features.extend([
                np.mean(hist),
                np.std(hist),
                skew(hist.flatten()),
                kurtosis(hist.flatten())
            ])
        return features
    
    def _texture_features(self, gray_img):
        """Texture analysis using LBP"""
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(gray_img, n_points, radius, method='uniform')
        hist, _ = np.histogram(lbp, bins=30, range=(0, 30))
        hist = hist / hist.sum()
        return list(hist)
    
    def _block_features(self, gray_img):
        """Block-based copy-move features"""
        h, w = gray_img.shape
        if h % self.block_size != 0 or w % self.block_size != 0:
            gray_img = gray_img[:h//self.block_size*self.block_size, 
                              :w//self.block_size*self.block_size]
            
        blocks = view_as_blocks(gray_img, (self.block_size, self.block_size))
        block_features = []
        
        for i in range(blocks.shape[0]):
            for j in range(blocks.shape[1]):
                block = blocks[i,j]
                block_features.extend([
                    np.mean(block),
                    np.var(block),
                    skew(block.flatten()),
                    cv2.Sobel(block, cv2.CV_64F, 1, 1).var()  # Edge content
                ])
                
        # Reduce dimensionality
        return [
            np.mean(block_features),
            np.std(block_features),
            skew(np.array(block_features)),
            kurtosis(np.array(block_features))
        ]
    
    def _tampering_features(self, img):
        """Forgery-specific artifacts"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. Double JPEG Compression Detection
        dct = cv2.dct(np.float32(gray)/255.0)
        dct_features = [
            np.mean(dct),
            np.var(dct),
            (dct > 0.1).sum()  # High-frequency components
        ]
        
        # 2. Edge Inconsistency
        edges = cv2.Canny(gray, 100, 200)
        edge_features = [
            np.mean(edges),
            (edges > 0).sum() / edges.size,  # Edge density
            cv2.Laplacian(gray, cv2.CV_64F).var()
        ]
        
        return dct_features + edge_features

# ============ USAGE EXAMPLE ============ 
if __name__ == "__main__":
    extractor = FeatureExtractor()
    
    # Test on sample images
    original_features = extractor.extract("dataset/original/img1.jpg")
    forged_features = extractor.extract("dataset/forged/img1_forged.jpg")
    
    print("Original features:", original_features[:10], "...")
    print("Forged features:", forged_features[:10], "...")
    
    # Feature difference analysis
    diff = np.abs(original_features - forged_features)
    print("\nMost distinguishing features:")
    for i in np.argsort(diff)[-5:][::-1]:
        print(f"Feature {i}: Diff={diff[i]:.2f} "
              f"(Original={original_features[i]:.2f}, "
              f"(Forged={forged_features[i]:.2f})")