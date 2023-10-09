import cv2
import numpy as np


def read_qr_code(filename):
    img = cv2.imread(filename)
    # grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
    # cv2.imshow('Image', blackAndWhiteImage)
    # cv2.imwrite('black.jpg', blackAndWhiteImage)
    # cv2.waitKey(0)
    detect = cv2.QRCodeDetector()
    link, corners_coord, _ = detect.detectAndDecode(img)
    corners_coord = corners_coord[0].tolist()
    qr_code_info = {
        'link': link,
        'top_left': (int(corners_coord[0][0]), int(corners_coord[0][1])),
        'top_right': (int(corners_coord[1][0]), int(corners_coord[1][1])),
        'bottom_right': (int(corners_coord[2][0]), int(corners_coord[2][1])),
        'bottom_left': (int(corners_coord[3][0]), int(corners_coord[3][1])),
        'middle_line': int(corners_coord[1][0] - corners_coord[0][0])  # value by OX
    }
    return qr_code_info, img


def crop_dashed_number(qr_code_info, img):
    crop_img = img[qr_code_info['bottom_right'][1] + 298:qr_code_info['bottom_right'][1] + 328,
               qr_code_info['top_left'][0] - 50:qr_code_info['top_right'][0] + 50]
    cv2.imshow('Image', crop_img)
    cv2.waitKey(0)
    return crop_img


def get_another_number(qr_code_info, img):  # TODO: HOW CALL IT ???
    crop_img = img[qr_code_info['bottom_right'][1] + 1398:qr_code_info['bottom_right'][1] + 1428,
               qr_code_info['top_left'][0] - 125:qr_code_info['top_right'][0] + 125]
    # cv2.imshow('Image', crop_img)
    # cv2.waitKey(0)

    template = cv2.imread('null2.png')
    w, h = template.shape[:-1]

    res = cv2.matchTemplate(crop_img, template, cv2.TM_CCOEFF_NORMED)
    threshold = .7
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):  # Switch columns and rows
        cv2.rectangle(crop_img, pt, (pt[0] + h, pt[1] + w), (0, 0, 255), 1)
    cv2.imshow('Image', crop_img)
    cv2.waitKey(0)
    return crop_img

# qr_code_info, img = read_qr_code('black.jpg') # 'images/l_258008_20230611085311.jpg')
# cropped_number = crop_dashed_number(qr_code_info, img)
# game_type = get_game_type(qr_code_info, img)
# another_number = get_another_number(qr_code_info, img)


numbers = {
  "0": [[0, 25], [0, 30]],
  "1": [[37, 60], [0, 30]],
  "2": [[70, 90], [0, 30]],
  "3": [[105, 125], [0, 30]],
  "4": [[135, 160], [0, 30]],
  "5": [[170, 195], [0, 30]],
  "6": [[205, 230], [0, 30]],
  "7": [[240, 260], [0, 30]],
  "8": [[270, 295], [0, 30]],
  "9": [[305, 330], [0, 30]]
}


# QRCode in check it is element relative to which we are looking for the remaining elements
class ParseCheck:
    def __init__(self, filename):
        self.filename = filename
        self.qr_code_info = {}
        self.original_img = cv2.imread(self.filename)
        grayImage = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2GRAY)
        thresh, self.img = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)

    def get_qr_code_link(self):
        detect = cv2.QRCodeDetector()
        link, corners_coord, _ = detect.detectAndDecode(self.img)
        corners_coord = corners_coord[0].tolist()
        self.qr_code_info = {
            'link': link,
            'top_left': (int(corners_coord[0][0]), int(corners_coord[0][1])),
            'top_right': (int(corners_coord[1][0]), int(corners_coord[1][1])),
            'bottom_right': (int(corners_coord[2][0]), int(corners_coord[2][1])),
            'bottom_left': (int(corners_coord[3][0]), int(corners_coord[3][1])),
            'middle_line': int(corners_coord[1][0] - corners_coord[0][0])  # value by OX
        }
        return link

    def get_game_type(self):
        crop_img = self.img[self.qr_code_info['bottom_right'][1] + 328:self.qr_code_info['bottom_right'][1] + 528,
                   self.qr_code_info['top_left'][0] - 158:self.qr_code_info['top_right'][0] + 158]
        # cv2.imshow('Image', crop_img)
        # cv2.waitKey(0)
        title = self.get_value_from_image(crop_img, 'title')
        return title

    def get_all_info(self):
        """
        check_info = {
            'qr_code_link': ...,
            'date': ...,
            'game_type': ...,
            'game_id': ...,
            'spent_on_ticket': ...,
            'dashed_number': ...,
            'spaced_number': ...,
            'cards': {
                'hearts': [],
                'diamonds': [],
                'spades': [],
                'clubs': []
            },

        }
        """
        self.get_qr_code_link()
        print(self.get_check_dashed_number())
        # print(self.get_check_spaced_number())
        # print(self.qr_code_info)
        # return self.get_game_type()

    def get_check_dashed_number(self):
        crop_img = self.img[self.qr_code_info['bottom_right'][1] + 298:self.qr_code_info['bottom_right'][1] + 328,
                   self.qr_code_info['top_left'][0] - 50:self.qr_code_info['top_right'][0] + 50]
        # cv2.imshow('Image', crop_img)
        # cv2.waitKey(0)
        dashed_number = self.get_value_from_image(crop_img, 'numbers')
        return dashed_number

    def get_check_spaced_number(self):
        crop_img = self.img[self.qr_code_info['bottom_right'][1] + 1398:self.qr_code_info['bottom_right'][1] + 1428,
                   self.qr_code_info['top_left'][0] - 125:self.qr_code_info['top_right'][0] + 125]

        spaced_number = self.get_value_from_image(crop_img, 'numbers')
        return spaced_number

    def get_cards(self):
        pass

    def get_date(self):
        pass

    def get_game_id(self):
        pass

    def get_value_from_image(self, croped_img, type=None):  # type can be numbers, cards, title
        edges = cv2.Canny(croped_img, 50, 200)
        # cv2.imshow('', edges)

        contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Переводим границы в кортеж
        info_of_contours = [cv2.boundingRect(contour) for contour in contours]  # (x, y, w, h)
        # Выстраиваем элементы в правильном порядке
        sorted_contours = sorted(info_of_contours, key=lambda contour: contour[0])
        # Убираем мусор
        sorted_contours = [contour for contour in sorted_contours if contour[2] >= 5 and contour[3] >= 15]
        # Убираем дубликаты элементов(если такие есть)
        for index, _ in enumerate(sorted_contours):
            try:
                is_duplicate = (sorted_contours[index + 1][0] - sorted_contours[index][0]) <= 5
                if is_duplicate:
                    current_square = sorted_contours[index][2] * sorted_contours[index][3]
                    next_square = sorted_contours[index + 1][2] * sorted_contours[index + 1][3]
                    if current_square > next_square:
                        del sorted_contours[index + 1]
                    else:
                        del sorted_contours[index + 1]
            except:
                pass
        result = ''
        for contour in sorted_contours:
            x, y, _, _ = contour
            cropped_contour = croped_img[y-2:y + 24, x-2:x + 15]
            # w = cropped_contour.shape[0]
            # h = cropped_contour.shape[1]

            if type == 'numbers':
                img = cv2.imread('numbers_and_letters.png')
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                res = cv2.matchTemplate(img, cropped_contour, cv2.TM_CCOEFF_NORMED)
                threshold = .7
                loc = np.where(res >= threshold)
                print(loc[::-1])  # TODO: Need get the best variant. HOW ?
                for pt in zip(*loc[::-1]):  # Switch columns and rows
                    # print(pt)
                    for key, value in numbers.items():
                        if pt[0] in range(*value[0]) and pt[1] in range(*value[1]):
                            result += key
                    break
                    # cv2.rectangle(img, pt, (pt[0] + h, pt[1] + w), (0, 0, 255), 1)
                # cv2.imshow('Image', img)
                # cv2.imshow('Image', cropped_contour)
                # cv2.waitKey(0)
            elif type == 'title':
                pass
            elif type == 'cards':
                pass
        print(result)
        return result

# TODO: need return
"""
check_info = {
    'qr_code_link': ...,
    'date': ...,
    'game_type': ...,
    'game_id': ...,
    'spent_on_ticket': ...,
    'dashed_number': ...,
    'spaced_number': ...,
    'cards': {
        'hearts': [], 
        'diamonds': [], 
        'spades': [], 
        'clubs': []
    },
    

}
"""

info = ParseCheck('images/l_258008_20230615175539.jpg').get_all_info()
