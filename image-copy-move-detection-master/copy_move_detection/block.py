import numpy as np
from sklearn.decomposition import PCA

class Blocks(object):
    """
    Contains a single image block and handle the calculation of characteristic features
    """

    def __init__(self, grayscale_image_block, rgb_image_block, x_coordinate, y_coordinate, block_dimension):
        """
        Initializing the input image
        :param grayscale_image_block: grayscale image block
        :param rgb_image_block: rgb image block
        :param x_coordinate: x coordinate (upper-left)
        :param y_coordinate: y coordinate (upper-left)
        :return: None
        """
        self.image_grayscale = grayscale_image_block  # block of grayscale image
        self.image_grayscale_pixels = self.image_grayscale.load()

        if rgb_image_block is not None:
            self.image_rgb = rgb_image_block
            self.image_rgb_pixels = self.image_rgb.load()
            self.is_image_rgb = True
        else:
            self.is_image_rgb = False

        self.coordinate = (x_coordinate, y_coordinate)
        self.block_dimension = block_dimension

    def compute_block(self):
        """
        Create a representation of the image block
        :return: image block representation data
        """
        block_data_list = []
        block_data_list.append(self.coordinate)
        block_data_list.append(self.compute_characteristic_features(precision=4))
        block_data_list.append(self.compute_pca(precision=6))
        return block_data_list

    def compute_pca(self, precision):
        """
        Compute Principal Component Analysis from the image block
        :param precision: characteristic features precision
        :return: Principal Component from the image block
        """
        pca_module = PCA(n_components=1)
        if self.is_image_rgb:
            image_array = np.array(self.image_rgb)  # shape (block_dim, block_dim, 3)
            flattened = image_array.reshape(-1, 3)  # shape (block_dim*block_dim, 3)
            pca_module.fit(flattened)
            principal_components = pca_module.components_
        else:
            image_array = np.array(self.image_grayscale)  # shape (block_dim, block_dim)
            flattened = image_array.reshape(-1, 1)  # shape (block_dim*block_dim, 1)
            pca_module.fit(flattened)
            principal_components = pca_module.components_

        precise_result = [round(element, precision) for element in principal_components.flatten()]
        return precise_result

    def compute_characteristic_features(self, precision):
        """
        Compute 7 characteristic features from every image blocks
        :param precision: feature characteristic precision
        :return: list of characteristic features rounded to precision
        """

        characteristic_feature_list = []

        # variables to compute characteristic features
        c4_part1 = 0
        c4_part2 = 0
        c5_part1 = 0
        c5_part2 = 0
        c6_part1 = 0
        c6_part2 = 0
        c7_part1 = 0
        c7_part2 = 0

        half_dim = self.block_dimension // 2  # integer division for indexing

        """ Compute c1, c2, c3 according to the image block's colorspace """

        if self.is_image_rgb:
            sum_of_red_pixel_value = 0
            sum_of_green_pixel_value = 0
            sum_of_blue_pixel_value = 0
            for y_coordinate in range(self.block_dimension):  # compute sum of the pixel value
                for x_coordinate in range(self.block_dimension):
                    tmp_red, tmp_green, tmp_blue = self.image_rgb_pixels[x_coordinate, y_coordinate]
                    sum_of_red_pixel_value += tmp_red
                    sum_of_green_pixel_value += tmp_green
                    sum_of_blue_pixel_value += tmp_blue

            sum_of_pixels = self.block_dimension * self.block_dimension
            sum_of_red_pixel_value /= sum_of_pixels  # mean from each of the colorspaces
            sum_of_green_pixel_value /= sum_of_pixels
            sum_of_blue_pixel_value /= sum_of_pixels

            characteristic_feature_list.append(sum_of_red_pixel_value)
            characteristic_feature_list.append(sum_of_green_pixel_value)
            characteristic_feature_list.append(sum_of_blue_pixel_value)

        else:
            characteristic_feature_list.extend([0, 0, 0])

        """ Compute c4, c5, c6 and c7 according to the pattern rule """

        for y_coordinate in range(self.block_dimension):  # compute the part 1 and part 2 of each feature characteristic
            for x_coordinate in range(self.block_dimension):
                pixel_val = self.image_grayscale_pixels[x_coordinate, y_coordinate]

                # c4: split horizontally at half_dim
                if y_coordinate <= half_dim:
                    c4_part1 += pixel_val
                else:
                    c4_part2 += pixel_val

                # c5: split vertically at half_dim
                if x_coordinate <= half_dim:
                    c5_part1 += pixel_val
                else:
                    c5_part2 += pixel_val

                # c6: split by diagonal x - y >= 0
                if x_coordinate - y_coordinate >= 0:
                    c6_part1 += pixel_val
                else:
                    c6_part2 += pixel_val

                # c7: split by anti-diagonal x + y <= block_dimension
                if x_coordinate + y_coordinate <= self.block_dimension:
                    c7_part1 += pixel_val
                else:
                    c7_part2 += pixel_val

        characteristic_feature_list.append(float(c4_part1) / float(c4_part1 + c4_part2))
        characteristic_feature_list.append(float(c5_part1) / float(c5_part1 + c5_part2))
        characteristic_feature_list.append(float(c6_part1) / float(c6_part1 + c6_part2))
        characteristic_feature_list.append(float(c7_part1) / float(c7_part1 + c7_part2))

        precise_result = [round(element, precision) for element in characteristic_feature_list]
        return precise_result
