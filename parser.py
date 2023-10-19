import os

import cv2
import numpy as np

cards = {
  # "ace-": [[], []],
  "king-diamonds": [[340, 480], [300, 500]],
  # "queen-": [[], []],
  # "jack-": [[], []],
  "ten-diamonds": [[340, 480], [40, 250]],

  # "ace-": [[], []],
  "king-spades": [[0, 180], [40, 250]],
  # "queen-": [[], []],
  # "jack-": [[], []],
  # "ten-": [[], []],
  "nine-spades": [[0, 180], [300, 500]],


  # "ace-": [[], []],
  # "king-": [[], []],
  # "queen-": [[], []],
  "jack-clubs": [[490, 630], [40, 250]],
  # "ten-": [[], []],
  "seven-clubs": [[490, 630], [300, 500]],

  "ace-hearts": [[190, 330], [40, 250]],
  # "king-": [[], []],
  "queen-hearts": [[190, 330], [300, 500]],
  # "jack-": [[], []],
  # "ten-": [[], []],
}

d_numbers = {
  "0": [[0, 25], [0, 30]],
  "1": [[37, 60], [0, 30]],
  "2": [[70, 90], [0, 30]],
  "3": [[100, 125], [0, 30]],
  "4": [[135, 160], [0, 30]],
  "5": [[170, 195], [0, 30]],
  "6": [[205, 230], [0, 30]],
  "7": [[240, 260], [0, 30]],
  "8": [[270, 295], [0, 30]],
  "9": [[305, 330], [0, 30]],
}

d_letters = {
  "B": [[0, 25], [40, 80]],
  "Q": [[35, 60], [40, 80]],
  "P": [[75, 100], [40, 80]],
  "V": [[110, 135], [40, 80]],
  "L": [[150, 170], [40, 80]],
  "C": [[185, 210], [40, 80]],
}

d_special_symbols = {
  "&": [[0, 25], [90, 115]]
}

d_all_symbols = {}
d_all_symbols.update(d_numbers)
d_all_symbols.update(d_letters)
d_all_symbols.update(d_special_symbols)


# QRCode in check it is element relative to which we are looking for the remaining elements
class ParseCheck:
    def __init__(self, img_path):
        self.img_path = img_path
        self.filename = os.path.basename(img_path).split('.')[0]
        self.qr_code_info = {}
        self.original_img = cv2.imread(self.img_path)
        self.check_info = {
            'qr_code_link': '',
            'date': '',
            'game_type': '',
            'game_id': 0,
            'spent_on_ticket': 0.0,
            'dashed_number': '',
            'spaced_number': '',
            'cards': {
                'hearts': [],
                'diamonds': [],
                'spades': [],
                'clubs': []
            },

        }
        '''
        if self.prep_ticket_out_brightness != 0:
            if self.prep_ticket_out_brightness > 0:
                shadow = self.prep_ticket_out_brightness
                highlight = 255
            else:
                shadow = 0
                highlight = 255 + self.prep_ticket_out_brightness
            alpha_b = (highlight - shadow) / 255
            gamma_b = shadow

            self.pic_crop = cv2.addWeighted(self.pic_crop, alpha_b, self.pic_crop, 0, gamma_b)
            self.pic_big_gray = cv2.addWeighted(self.pic_big_gray, alpha_b, self.pic_big_gray, 0, gamma_b)
            self.pic_big_color = cv2.addWeighted(self.pic_big_color, alpha_b, self.pic_big_color, 0, gamma_b)

        if self.prep_ticket_out_contrast != 0:
            f = 131 * (self.prep_ticket_out_contrast + 127) / (127 * (131 - self.prep_ticket_out_contrast))
            alpha_c = f
            gamma_c = 127 * (1 - f)

            self.pic_crop = cv2.addWeighted(self.pic_crop, alpha_c, self.pic_crop, 0, gamma_c)
            self.pic_big_gray = cv2.addWeighted(self.pic_big_gray, alpha_c, self.pic_big_gray, 0, gamma_c)
            self.pic_big_color = cv2.addWeighted(self.pic_big_color, alpha_c, self.pic_big_color, 0, gamma_c)
        '''
        grayImage = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2GRAY)
        thresh, self.img = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)

    @staticmethod
    def _save_debug_img(img, filename):
        if not os.path.exists("debug_img"):
            os.mkdir("debug_img")
        path = os.path.join("debug_img", filename)
        cv2.imwrite(path, img)

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
        self._save_debug_img(crop_img, f"cropped_game_type({self.filename}).jpg")
        title = self.get_value_from_image(crop_img, 'title')
        return title

    def get_all_info(self):
        self.get_qr_code_link()
        print("Dashed number:", self.get_check_dashed_number())
        print("Spaced number:", self.get_check_spaced_number())
        print("Game id:", self.get_game_id())
        print("Date:", self.get_date())
        print("Cards:", self.get_cards())
        print("Spent money:", self.get_spent_money())
        print("Game type:", self.get_game_type())
        # print(self.qr_code_info)
        return self.check_info

    def get_spent_money(self):
        crop_img = self.img[self.qr_code_info['bottom_right'][1] + 1300:self.qr_code_info['bottom_right'][1] + 1330,
                            self.qr_code_info['top_left'][0] - 125:self.qr_code_info['top_left'][0] + 75]
        self._save_debug_img(crop_img, f"cropped_spent_money_bf_blur({self.filename}).jpg")
        crop_img = cv2.GaussianBlur(crop_img, [3, 3], 0)
        self._save_debug_img(crop_img, f"cropped_spent_money_af_blur({self.filename}).jpg")
        parsed_numbers = self.get_value_from_image(crop_img, 'numbers', parse_just_in_numbers=True)

        numbers = ''
        for symbol in parsed_numbers:
            if symbol.isnumeric():
                numbers += symbol
        return int(numbers[0:len(numbers)-2])  # TODO: int or float ???

    def get_check_dashed_number(self):
        crop_img = self.img[self.qr_code_info['bottom_right'][1] + 298:self.qr_code_info['bottom_right'][1] + 328,
                            self.qr_code_info['top_left'][0] - 50:self.qr_code_info['top_right'][0] + 50]
        self._save_debug_img(crop_img, f"cropped_dashed_number_bf_blur({self.filename}).jpg")
        crop_img = cv2.GaussianBlur(crop_img, (3, 3), 0)
        self._save_debug_img(crop_img, f"cropped_dashed_number_af_blur({self.filename}).jpg")
        numbers = self.get_value_from_image(crop_img, 'numbers', parse_just_in_numbers=True)
        if len(numbers) == 19:
            dashed_number = f"{numbers[:4]}-{numbers[4:13]}-{numbers[13:]}"
            return dashed_number
        else:
            raise Exception(f"The length of the number is not correct(dashed_number)\nNumber: {numbers}")

    def get_check_spaced_number(self):
        crop_img = self.img[self.qr_code_info['bottom_right'][1] + 1398:self.qr_code_info['bottom_right'][1] + 1428,
                            self.qr_code_info['top_left'][0] - 125:self.qr_code_info['top_right'][0] + 125]
        self._save_debug_img(crop_img, f"cropped_spaced_number_bf_blur({self.filename}).jpg")
        crop_img = cv2.GaussianBlur(crop_img, (3, 3), 0)
        self._save_debug_img(crop_img, f"cropped_spaced_number_af_blur({self.filename}).jpg")
        numbers = self.get_value_from_image(crop_img, 'numbers')
        if len(numbers) == 26:
            spaced_number = f"{numbers[:8]} {numbers[8:17]} {numbers[17:]}"
            return spaced_number
        else:
            raise Exception(f"The length of the number is not correct(spaced_number)\nNumber: {numbers}")

    def get_cards(self):
        crop_img = self.img[self.qr_code_info['bottom_right'][1] + 780:self.qr_code_info['bottom_right'][1] + 1000,
                            self.qr_code_info['top_left'][0] - 200:self.qr_code_info['top_right'][0] + 200]
        self._save_debug_img(crop_img, f"cropped_cards({self.filename}).jpg")
        cards = self.get_value_from_image(crop_img, 'cards')
        return cards

    def get_date(self):
        crop_img = self.img[self.qr_code_info['bottom_right'][1] + 1115:self.qr_code_info['bottom_right'][1] + 1145,
                            self.qr_code_info['top_left'][0] - 85:self.qr_code_info['top_left'][0] + 245]
        self._save_debug_img(crop_img, f"cropped_date_bf_blur({self.filename}).jpg")
        # crop_img = cv2.GaussianBlur(crop_img, (3, 3), 0)
        # self._save_debug_img(crop_img, f"cropped_date_af_blur({self.filename}).jpg")
        numbers = self.get_value_from_image(crop_img, 'numbers', parse_just_in_numbers=True)
        if len(numbers) == 12:
            return f"{numbers[:2]}:{numbers[2:4]}:{numbers[4:6]} {numbers[6:8]}.{numbers[8:10]}.{numbers[10:]}"
        else:
            raise Exception(f"The length of the number is not correct(dashed_number)\nNumber: {numbers}")

    def get_game_id(self):
        crop_img = self.img[self.qr_code_info['bottom_left'][1] + 1075:self.qr_code_info['bottom_left'][1] + 1115,
                            self.qr_code_info['top_left'][0] - 120:self.qr_code_info['top_left'][0] + 90]
        self._save_debug_img(crop_img, f"cropped_game_id({self.filename}).jpg")
        numbers = self.get_value_from_image(crop_img, 'numbers', parse_just_in_numbers=True)
        return numbers

    @staticmethod
    def get_value_from_image(cropped_img, data_type, parse_just_in_numbers=False):  # type can be numbers, cards, title, date
        min_width_and_height = {  # For cropped elements
            "numbers": [4, 11],
            "cards": [100, 150],
            "title": [1000, 1000],  # this need find like all image
            "date": [4, 11]
        }
        edges = cv2.Canny(cropped_img, 50, 200)
        # cv2.imshow('', edges)
        # cv2.waitKey(0)
        contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Переводим границы в кортеж
        info_of_contours = [cv2.boundingRect(contour) for contour in contours]  # (x, y, w, h)
        # Выстраиваем элементы в правильном порядке
        sorted_contours = sorted(info_of_contours, key=lambda contour: contour[0])
        # Убираем мусор
        sorted_contours = [contour for contour in sorted_contours if contour[2] >= min_width_and_height[data_type][0] and contour[3] >= min_width_and_height[data_type][1]]
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
            except Exception as ex:
                print(f"Were caught exception - {ex}")
        result = ''
        for contour in sorted_contours:
            if data_type == "cards":
                x, y, w, h = contour
                cropped_contour = cropped_img[y-2:y + h, x-2:x + w]
            else:
                x, y, _, _ = contour
                cropped_contour = cropped_img[y:y + 30, x:x + 16]
            ret3, cropped_contour = cv2.threshold(cropped_contour, 170, 255, cv2.THRESH_BINARY)

            if data_type == 'numbers':
                if parse_just_in_numbers:
                    img = cv2.imread('numbers.png')
                else:
                    img = cv2.imread('numbers_and_letters.png')
                    # print(cropped_contour)
                    # if cropped_contour is not None:
                    #     cv2.imshow('Image', cropped_contour)
                    #     cv2.waitKey(0)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # TODO: border it is bad idea
                # cropped_contour = cv2.copyMakeBorder(
                #     cropped_contour,
                #     top=1,
                #     bottom=1,
                #     left=1,
                #     right=1,
                #     borderType=cv2.BORDER_CONSTANT,
                #     value=[255, 255, 255]
                # )
                if cropped_contour is not None:
                    res = cv2.matchTemplate(img, cropped_contour, cv2.TM_CCOEFF_NORMED)
                    y, x = np.unravel_index(res.argmax(), res.shape)

                    for key, value in d_all_symbols.items():
                        if x in range(*value[0]) and y in range(*value[1]):
                            result += key
            elif data_type == 'title':
                pass
                # img = cv2.imread('titles.png')
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # res = cv2.matchTemplate(img, cropped_contour, cv2.TM_CCOEFF_NORMED)
                # threshold = .7
                # loc = np.where(res >= threshold)
                # # print(loc[::-1])  # TODO: Need get the best variant. HOW ?
                # for pt in zip(*loc[::-1]):  # Switch columns and rows
                #     # print(pt)
                #     for key, value in cards.items():
                #         if pt[0] in range(*value[0]) and pt[1] in range(*value[1]):
                #             result += f"{key} "
                #     break
            elif data_type == 'cards':
                img = cv2.imread('cards.png')
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                res = cv2.matchTemplate(img, cropped_contour, cv2.TM_CCOEFF_NORMED)
                threshold = .7
                loc = np.where(res >= threshold)
                for pt in zip(*loc[::-1]):  # Switch columns and rows
                    for key, value in cards.items():
                        if pt[0] in range(*value[0]) and pt[1] in range(*value[1]):
                            result += f"{key} "
                    break
        return result


# TODO: Добавить контраста и яркости

# TODO: Нужны новые чеки для анализа (чтобы вытянуть оттуда новые буквы, карты, типы игр???)
# TODO: Game id, what in the brackets???

print("First check")
info = ParseCheck('images/l_258008_20230615175539.jpg').get_all_info()

print("Second check")
info1 = ParseCheck('images/l_258011_20230615175608.jpg').get_all_info()


"""
if self.prep_ticket_out_brightness != 0:
                    if self.prep_ticket_out_brightness > 0:
                        shadow = self.prep_ticket_out_brightness
                        highlight = 255
                    else:
                        shadow = 0
                        highlight = 255 + self.prep_ticket_out_brightness
                    alpha_b = (highlight - shadow) / 255
                    gamma_b = shadow

                    self.pic_crop = cv2.addWeighted(self.pic_crop, alpha_b, self.pic_crop, 0, gamma_b)
                    self.pic_big_gray = cv2.addWeighted(self.pic_big_gray, alpha_b, self.pic_big_gray, 0, gamma_b)
                    self.pic_big_color = cv2.addWeighted(self.pic_big_color, alpha_b, self.pic_big_color, 0, gamma_b)

                if self.prep_ticket_out_contrast != 0:
                    f = 131 * (self.prep_ticket_out_contrast + 127) / (127 * (131 - self.prep_ticket_out_contrast))
                    alpha_c = f
                    gamma_c = 127 * (1 - f)

                    self.pic_crop = cv2.addWeighted(self.pic_crop, alpha_c, self.pic_crop, 0, gamma_c)
                    self.pic_big_gray = cv2.addWeighted(self.pic_big_gray, alpha_c, self.pic_big_gray, 0, gamma_c)
                    self.pic_big_color = cv2.addWeighted(self.pic_big_color, alpha_c, self.pic_big_color, 0, gamma_c)
"""
