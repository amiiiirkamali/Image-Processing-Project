import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def create_directory(dir_name):
    """ایجاد دایرکتوری در صورت عدم وجود"""
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def show_images(images, titles, rows, cols, figsize=(15, 10)):
    """نمایش تصاویر در یک شبکه"""
    fig = plt.figure(figsize=figsize)
    for i in range(len(images)):
        ax = fig.add_subplot(rows, cols, i + 1)
        if len(images[i].shape) == 2:
            ax.imshow(images[i], cmap='gray')
        else:
            ax.imshow(images[i])
        ax.set_title(titles[i])
        ax.axis('off')
    plt.tight_layout()
    return fig


def save_figure(fig, filename):
    """ذخیره شکل در فایل"""
    fig.savefig(filename, bbox_inches='tight')
    plt.close(fig)


# فیلترهای پردازش تصویر
def negative(image):
    """معکوس کردن مقادیر پیکسل‌ها (نگاتیو)"""
    return 255 - image


def sobel_edge_detection(image):
    """لبه‌یابی با فیلتر سوبل"""
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobelx ** 2 + sobely ** 2)
    sobel = np.uint8(255 * sobel / np.max(sobel))
    return sobel


def histogram_equalization(image):
    """همسان‌سازی هیستوگرام"""
    return cv2.equalizeHist(image)


def gaussian_blur(image, kernel_size=7):
    """فیلتر میانگین گیری گوسی"""
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def threshold(image, thresh=128):
    """اعمال حد آستانه"""
    _, binary = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)
    return binary


def median_filter(image, kernel_size=5):
    """فیلتر غیرخطی میانه"""
    return cv2.medianBlur(image, kernel_size)


def roberts_edge_detection(image):
    """لبه‌یابی با ماسک روبرت"""
    kernel_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
    kernel_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)

    x = cv2.filter2D(image.astype(np.float32), -1, kernel_x)
    y = cv2.filter2D(image.astype(np.float32), -1, kernel_y)

    return np.uint8(np.clip(np.abs(x) + np.abs(y), 0, 255))


def salt_pepper_noise(image, salt_prob=0.025, pepper_prob=0.025):
    """افزودن نویز فلفل نمکی"""
    noisy = np.copy(image)
    # نویز نمک (سفید)
    num_salt = np.ceil(salt_prob * image.size)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy[tuple(coords)] = 255

    # نویز فلفل (سیاه)
    num_pepper = np.ceil(pepper_prob * image.size)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy[tuple(coords)] = 0

    return noisy


def log_transform(image, c=1):
    """تبدیل لگاریتمی"""
    # افزودن 1 برای جلوگیری از خطای لگاریتم صفر
    log_image = c * np.log1p(image.astype(np.float32))
    # نرمال‌سازی به محدوده 0-255
    log_image = np.uint8(255 * log_image / np.max(log_image))
    return log_image


def maximum_filter(image, kernel_size=3):
    """فیلتر ماکزیمم"""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(image, kernel)


def minimum_filter(image, kernel_size=3):
    """فیلتر مینیمم"""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.erode(image, kernel)


def gaussian_noise(image, mean=0, sigma=15):
    """افزودن نویز گوسی"""
    row, col = image.shape
    gauss = np.random.normal(mean, sigma, (row, col))
    gauss = gauss.reshape(row, col)
    noisy = np.clip(image + gauss, 0, 255).astype(np.uint8)
    return noisy


def box_blur(image, kernel_size=3):
    """فیلتر میانگین گیری باکس"""
    return cv2.blur(image, (kernel_size, kernel_size))


def custom_edge_detection1(image):
    """فیلتر لبه‌یابی با ماسک دلخواه برای برگ گیاه"""
    kernel = np.array([[-1, -1, -1],
                       [-1, 8, -1],
                       [-1, -1, -1]], dtype=np.float32)

    return np.uint8(np.clip(cv2.filter2D(image, -1, kernel), 0, 255))


def custom_edge_detection2(image):
    """فیلتر لبه‌یابی با ماسک دلخواه برای تصویر میکروسکوپی"""
    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]], dtype=np.float32)

    return np.uint8(np.clip(cv2.filter2D(image, -1, kernel), 0, 255))


def binary_threshold(image, lower=100, upper=200):
    """اعمال حد آستانه دو سطحی"""
    result = np.zeros_like(image)

    # پیکسل‌های کمتر از lower سیاه (0)
    # پیکسل‌های بین lower و upper خاکستری (128)
    # پیکسل‌های بیشتر از upper سفید (255)
    result[image < lower] = 0
    result[(image >= lower) & (image <= upper)] = 128
    result[image > upper] = 255

    return result


def power_transform(image, gamma=2.2):
    """تبدیل توانی (گاما)"""
    # نرمال‌سازی تصویر به محدوده 0-1
    normalized = image / 255.0
    # اعمال تبدیل توانی
    result = np.power(normalized, gamma)
    # تبدیل به محدوده 0-255
    return np.uint8(result * 255)


def linear_transform(image, brightness=30):
    """تبدیل خطی (تغییر روشنایی)"""
    result = np.clip(image.astype(np.int16) + brightness, 0, 255).astype(np.uint8)
    return result


def process_images():
    """پردازش تصاویر و ذخیره نتایج"""
    # ایجاد دایرکتوری‌های خروجی
    output_dir = "output"
    individual_filters_dir = os.path.join(output_dir, "individual_filters")
    sequential_filters_dir = os.path.join(output_dir, "sequential_filters")

    create_directory(output_dir)
    create_directory(individual_filters_dir)
    create_directory(sequential_filters_dir)

    # لیست فیلترها برای هر تصویر
    image_filters = [
        # تصویر 1: چهره انسان
        [
            {"func": negative, "name": "Negative"},
            {"func": sobel_edge_detection, "name": "Sobel Edge Detection"},
            {"func": histogram_equalization, "name": "Histogram Equalization"}
        ],
        # تصویر 2: منظره طبیعی
        [
            {"func": gaussian_blur, "name": "Gaussian Blur"},
            {"func": threshold, "name": "Threshold"},
            {"func": median_filter, "name": "Median Filter"}
        ],
        # تصویر 3: ساختمان شهری
        [
            {"func": roberts_edge_detection, "name": "Roberts Edge Detection"},
            {"func": salt_pepper_noise, "name": "Salt & Pepper Noise"},
            {"func": log_transform, "name": "Logarithmic Transform"}
        ],
        # تصویر 4: بافت سنگ
        [
            {"func": maximum_filter, "name": "Maximum Filter"},
            {"func": minimum_filter, "name": "Minimum Filter"},
            {"func": gaussian_noise, "name": "Gaussian Noise"}
        ],
        # تصویر 5: برگ گیاه
        [
            {"func": histogram_equalization, "name": "Histogram Equalization"},
            {"func": box_blur, "name": "Box Blur"},
            {"func": custom_edge_detection1, "name": "Custom Edge Detection"}
        ],
        # تصویر 6: اثر انگشت
        [
            {"func": lambda img: threshold(img, 150), "name": "Threshold (150)"},
            {"func": lambda img: median_filter(img, 5), "name": "Median Filter (5x5)"},
            {"func": negative, "name": "Negative"}
        ],
        # تصویر 7: آسمان ابری
        [
            {"func": power_transform, "name": "Power Transform (gamma=2.2)"},
            {"func": lambda img: gaussian_blur(img, 7), "name": "Gaussian Blur (7x7)"},
            {"func": lambda img: salt_pepper_noise(img, 0.025, 0.025), "name": "Salt & Pepper Noise (5%)"}
        ],
        # تصویر 8: بافت پارچه
        [
            {"func": sobel_edge_detection, "name": "Sobel Edge Detection"},
            {"func": histogram_equalization, "name": "Histogram Equalization"},
            {"func": lambda img: box_blur(img, 5), "name": "Box Blur (5x5)"}
        ],
        # تصویر 9: گل‌های مرداب
        [
            {"func": lambda img: gaussian_noise(img, 0, 10), "name": "Gaussian Noise (σ=10)"},
            {"func": lambda img: median_filter(img, 3), "name": "Median Filter (3x3)"},
            {"func": lambda img: linear_transform(img, 30), "name": "Linear Transform (+30)"}
        ],
        # تصویر 10: تصویر میکروسکوپی
        [
            {"func": lambda img: binary_threshold(img, 100, 200), "name": "Binary Threshold (100, 200)"},
            {"func": custom_edge_detection2, "name": "Custom Edge Detection"},
            {"func": negative, "name": "Negative"}
        ]
    ]

    # تصاویر منتخب برای فیلترهای متوالی (شماره‌های 0، 3، 5، 8 معادل تصاویر 1، 4، 6، 9)
    selected_images = [0, 3, 5, 8]

    # بخش اول: اعمال فیلترهای مجزا روی 10 تصویر
    for i in range(1, 11):
        # بارگذاری تصویر
        image_path = os.path.join("images", f"image{i}.jpg")
        if not os.path.exists(image_path):
            print(f"تصویر {image_path} یافت نشد!")
            continue

        # خواندن تصویر و تبدیل به سطح خاکستری
        image = cv2.imread(image_path)
        if image is None:
            print(f"خطا در خواندن تصویر {image_path}!")
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # اعمال فیلترهای مجزا
        filters = image_filters[i - 1]
        filtered_images = [gray]
        titles = [f"Original Image {i}"]

        for filter_info in filters:
            filtered = filter_info["func"](gray)
            filtered_images.append(filtered)
            titles.append(filter_info["name"])

        # نمایش و ذخیره نتایج
        fig = show_images(filtered_images, titles, 1, 4)
        save_figure(fig, os.path.join(individual_filters_dir, f"image{i}_filters.png"))

        print(f"فیلترهای مجزا برای تصویر {i} اعمال شدند.")

    # بخش دوم: اعمال فیلترهای متوالی روی 4 تصویر منتخب
    for idx in selected_images:
        i = idx + 1  # شماره تصویر

        # بارگذاری تصویر
        image_path = os.path.join("images", f"image{i}.jpg")
        if not os.path.exists(image_path):
            print(f"تصویر {image_path} یافت نشد!")
            continue

        # خواندن تصویر و تبدیل به سطح خاکستری
        image = cv2.imread(image_path)
        if image is None:
            print(f"خطا در خواندن تصویر {image_path}!")
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # فیلترهای مربوط به این تصویر
        filters = image_filters[idx]

        # ترتیب اول: 1→2→3
        result1 = gray.copy()
        intermediate1 = [gray]
        titles1 = ["Original"]

        for j, filter_info in enumerate(filters):
            result1 = filter_info["func"](result1)
            intermediate1.append(result1)
            titles1.append(f"Step {j + 1}: {filter_info['name']}")

        # ترتیب دوم: 3→2→1 (معکوس)
        result2 = gray.copy()
        intermediate2 = [gray]
        titles2 = ["Original"]

        for j, filter_info in enumerate(reversed(filters)):
            result2 = filter_info["func"](result2)
            intermediate2.append(result2)
            titles2.append(f"Step {j + 1}: {filters[2 - j]['name']}")

        # نمایش و ذخیره نتایج
        fig1 = show_images(intermediate1, titles1, 1, 5, figsize=(20, 5))
        save_figure(fig1, os.path.join(sequential_filters_dir, f"image{i}_sequential_123.png"))

        fig2 = show_images(intermediate2, titles2, 1, 5, figsize=(20, 5))
        save_figure(fig2, os.path.join(sequential_filters_dir, f"image{i}_sequential_321.png"))

        # مقایسه نتایج نهایی دو ترتیب
        comparison = [gray, intermediate1[-1], intermediate2[-1]]
        comparison_titles = ["Original", "Result with order 1→2→3", "Result with order 3→2→1"]

        fig3 = show_images(comparison, comparison_titles, 1, 3, figsize=(15, 5))
        save_figure(fig3, os.path.join(sequential_filters_dir, f"image{i}_comparison.png"))

        print(f"فیلترهای متوالی برای تصویر {i} با ترتیب‌های مختلف اعمال شدند.")


if __name__ == "__main__":
    print("شروع پردازش تصاویر...")
    process_images()
    print("پردازش تصاویر با موفقیت به پایان رسید. نتایج در پوشه output ذخیره شدند.")