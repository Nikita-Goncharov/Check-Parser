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

    def get_game_type(self):
        crop_img = self.img[self.qr_code_info['bottom_right'][1] + 328:self.qr_code_info['bottom_right'][1] + 528,
                   self.qr_code_info['top_left'][0] - 158:self.qr_code_info['top_right'][0] + 158]
        cv2.imshow('Image', crop_img)
        cv2.waitKey(0)
        return crop_img

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
        print(self.qr_code_info)
        return self.get_game_type()

    def get_check_dashed_number(self):
        crop_img = self.img[self.qr_code_info['bottom_right'][1] + 298:self.qr_code_info['bottom_right'][1] + 328,
                   self.qr_code_info['top_left'][0] - 50:self.qr_code_info['top_right'][0] + 50]
        cv2.imshow('Image', crop_img)
        cv2.waitKey(0)
        return crop_img

    def get_check_spaced_number(self):
        crop_img = self.img[self.qr_code_info['bottom_right'][1] + 1398:self.qr_code_info['bottom_right'][1] + 1428,
                   self.qr_code_info['top_left'][0] - 125:self.qr_code_info['top_right'][0] + 125]

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

    def get_cards(self):
        pass

    def get_date(self):
        pass

    def get_game_id(self):
        pass

    def get_value_from_image(self, croped_img, type=None):  # type can be numbers, cards, title
        edges = cv2.Canny(croped_img, 50, 200)
        cv2.imshow('', edges)

        contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(contours[0].data, contours[0].item, dir(contours[0]))
        # sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
        # ctrs = sorted(contours, key=cv2.contourArea, reverse=True)
        for c in contours:

            x, y, w, h = cv2.boundingRect(c)
            if w >= 5 and h >= 15:
                cropped_contour = self.img[y:y + 20, x:x + 15]
                w, h = cropped_contour.shape[:-1]
                numbers_and_letters = cv2.imread('numbers_and_letters.png')
                res = cv2.matchTemplate(numbers_and_letters, cropped_contour, cv2.TM_CCOEFF_NORMED)
                threshold = .7
                # print(dir(res))
                loc = np.where(res >= threshold)
                for pt in zip(*loc[::-1]):  # Switch columns and rows
                    cv2.rectangle(numbers_and_letters, pt, (pt[0] + h, pt[1] + w), (0, 0, 255), 1)
                cv2.imshow('Image', numbers_and_letters)
                # cv2.imshow('Image', cropped_contour)
                cv2.waitKey(0)



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
