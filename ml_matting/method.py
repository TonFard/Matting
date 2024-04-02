import pymatting


class BaseMLMatting(object):
    def __init__(self, alpha_estimator, **kargs):
        self.alpha_estimator = alpha_estimator
        self.kargs = kargs

    def __call__(self, image, trimap):
        image = self.__to_float64(image)
        trimap = self.__to_float64(trimap)
        alpha_matte = self.alpha_estimator(image, trimap, **self.kargs)
        return alpha_matte

    def __to_float64(self, x):
        x_dtype = x.dtype
        assert x_dtype in ["float32", "float64"]
        x = x.astype("float64")
        return x


class CloseFormMatting(BaseMLMatting):
    def __init__(self, **kargs):
        cf_alpha_estimator = pymatting.estimate_alpha_cf
        super().__init__(cf_alpha_estimator, **kargs)


class KNNMatting(BaseMLMatting):
    def __init__(self, **kargs):
        knn_alpha_estimator = pymatting.estimate_alpha_knn
        super().__init__(knn_alpha_estimator, **kargs)


class LearningBasedMatting(BaseMLMatting):
    def __init__(self, **kargs):
        lbdm_alpha_estimator = pymatting.estimate_alpha_lbdm
        super().__init__(lbdm_alpha_estimator, **kargs)


class FastMatting(BaseMLMatting):
    def __init__(self, **kargs):
        lkm_alpha_estimator = pymatting.estimate_alpha_lkm
        super().__init__(lkm_alpha_estimator, **kargs)


class RandomWalksMatting(BaseMLMatting):
    def __init__(self, **kargs):
        rw_alpha_estimator = pymatting.estimate_alpha_rw
        super().__init__(rw_alpha_estimator, **kargs)


if __name__ == "__main__":
    from pymatting.util.util import load_image, save_image, stack_images
    from estimate_foreground_ml import estimate_foreground_ml
    import cv2

    root = "/mnt/liuyi22/PaddlePaddle/PaddleSeg/Matting/data/examples/"
    image_path = root + "lemur.png"
    trimap_path = root + "lemur_trimap.png"
    cutout_path = root + "lemur_cutout.png"
    image = cv2.cvtColor(
        cv2.imread(image_path).astype("float64"), cv2.COLOR_BGR2RGB) / 255.0

    cv2.imwrite("image.png", (image * 255).astype('uint8'))
    trimap = load_image(trimap_path, "GRAY")
    print(image.shape, trimap.shape)
    print(image.dtype, trimap.dtype)
    cf = CloseFormMatting()
    alpha = cf(image, trimap)

    # alpha = pymatting.estimate_alpha_lkm(image, trimap)

    foreground = estimate_foreground_ml(image, alpha)

    cutout = stack_images(foreground, alpha)

    save_image(cutout_path, cutout)