import os

import cv2
import numpy as np
from qreader import QReader
from colorama import init, Fore, Back, Style

init(autoreset=True)

game_types = {
    "lotto": [[0, 0], [0, 0]],
    "123": [[0, 0], [0, 0]],
    "777": [[0, 0], [0, 0]],
    "chance": [[0, 700], [0, 250]]
}

cards = {
    "ace-spades": [[40, 185], [40, 260]],
    "king-spades": [[40, 185], [280, 505]],
    "queen-spades": [[40, 185], [515, 760]],
    "jack-spades": [[40, 185], [780, 1020]],
    "ten-spades": [[40, 185], [1040, 1280]],
    "nine-spades": [[40, 185], [1300, 1540]],
    "eighth-spades": [[40, 185], [1560, 1800]],
    "seven-spades": [[40, 185], [1820, 2060]],

    "ace-hearts": [[192, 335], [40, 260]],
    "king-hearts": [[192, 335], [280, 505]],
    "queen-hearts": [[192, 335], [515, 760]],
    "jack-hearts": [[192, 335], [780, 1020]],
    "ten-hearts": [[192, 335], [1040, 1280]],
    "nine-hearts": [[192, 335], [1300, 1540]],
    "eighth-hearts": [[192, 335], [1560, 1800]],
    "seven-hearts": [[192, 335], [1820, 2060]],


    "ace-diamonds": [[340, 485], [40, 260]],
    "king-diamonds": [[340, 485], [280, 505]],
    "queen-diamonds": [[340, 485], [515, 760]],
    "jack-diamonds": [[340, 485], [780, 1020]],
    "ten-diamonds": [[340, 485], [1040, 1280]],
    "nine-diamonds": [[340, 485], [1300, 1540]],
    "eighth-diamonds": [[340, 485], [1560, 1800]],
    "seven-diamonds": [[340, 485], [1820, 2060]],


    "ace-clubs": [[490, 640], [40, 260]],
    "king-clubs": [[490, 640], [280, 505]],
    "queen-clubs": [[490, 640], [515, 760]],
    "jack-clubs": [[490, 640], [780, 1020]],
    "ten-clubs": [[490, 640], [1040, 1280]],
    "nine-clubs": [[490, 640], [1300, 1540]],
    "eighth-clubs": [[490, 640], [1560, 1800]],
    "seven-clubs": [[490, 6400], [1820, 2060]],
}

d_numbers = {
    "0": [[0, 29], [0, 30]],
    "1": [[35, 65], [0, 30]],
    "2": [[70, 90], [0, 30]],
    "3": [[100, 125], [0, 30]],
    "4": [[135, 160], [0, 30]],
    "5": [[170, 195], [0, 30]],
    "6": [[205, 230], [0, 30]],
    "7": [[240, 265], [0, 30]],
    "8": [[270, 295], [0, 30]],
    "9": [[305, 330], [0, 30]],
}

d_letters = {
    "B": [[0, 25], [40, 80]],
    "Q": [[35, 65], [40, 80]],
    "P": [[75, 100], [40, 80]],
    "V": [[110, 135], [40, 80]],
    "L": [[150, 175], [40, 80]],
    "C": [[185, 210], [40, 80]],
    "S": [[215, 240], [40, 80]],
    "F": [[245, 270], [40, 80]],
    "Y": [[280, 305], [40, 80]],
    "G": [[310, 335], [40, 80]],
    "H": [[345, 370], [40, 80]],
    "J": [[375, 400], [40, 80]],
    "R": [[405, 430], [40, 80]],
    "W": [[435, 462], [40, 80]],
    "K": [[465, 490], [40, 80]],
    "D": [[495, 520], [40, 80]],
    "T": [[525, 552], [40, 80]],
    "N": [[555, 580], [40, 80]],
    "Z": [[585, 610], [40, 80]],
    "X": [[615, 640], [40, 80]],
    "M": [[645, 670], [40, 80]],
}

d_special_symbols = {
    "&": [[0, 30], [85, 120]],
    "%": [[35, 63], [85, 120]],
    "#": [[67, 90], [85, 120]],
    "@": [[95, 125], [85, 120]],
    "-": [[130, 160], [85, 120]],
    "*": [[165, 180], [85, 120]],
}

d_all_symbols = {}
d_all_symbols.update(d_numbers)
d_all_symbols.update(d_letters)
d_all_symbols.update(d_special_symbols)


# Content lines in check it is elements relative to which we are looking for the remaining elements
class GetCheckGameType:
    # Distance from lines to needed elements  (Does really need ???)
    games_elements_distance = {
        # "123_regular": "",
        "777_regular": {
            "spaced_number": [000, "bottom_line"],
            "dashed_number": [000, "bottom_line"],
            "date": [000, "bottom_line"],
            "game_id": [000, "bottom_line"],
            "sum": [000, "bottom_line"],
            "qr_code": [000, "top_line"],  # If not found
            "game_subtype": [000, "bottom_line"],  # from bottom to top
            "table": [[], [], []]  # in every element will 7 numbers
        },
        "777_systematic": {
            "spaced_number": [000, "bottom_line"],
            "dashed_number": [000, "bottom_line"],
            "date": [000, "bottom_line"],
            "game_id": [000, "bottom_line"],
            "sum": [000, "bottom_line"],
            "qr_code": [000, "top_line"],  # If not found
            "game_subtype": [000, "bottom_line"],  # from bottom to top
            "table": [[], [], []]  # will only first element filled with 8/9 numbers
        },
        "chance_regular": {
            "spaced_number": [000, "bottom_line"],
            "dashed_number": [000, "bottom_line"],
            "date": [000, "bottom_line"],
            "game_id": [000, "bottom_line"],
            "sum": [000, "bottom_line"],
            "qr_code": [000, "top_line"],  # If not found
            "game_subtype": [000, "bottom_line"],  # from bottom to top
            "cards": ""
        },
        "chance_multi": {
            "spaced_number": [000, "bottom_line"],
            "dashed_number": [000, "bottom_line"],
            "date": [000, "bottom_line"],
            "game_id": [000, "bottom_line"],
            "sum": [000, "bottom_line"],
            "qr_code": [000, "top_line"],  # If not found
            "game_subtype": [000, "bottom_line"],  # from bottom to top
            "cards": ""
        },
        "chance_systematic": {
            "spaced_number": [000, "bottom_line"],
            "dashed_number": [000, "bottom_line"],
            "date": [000, "bottom_line"],
            "game_id": [000, "bottom_line"],
            "sum": [000, "bottom_line"],
            "qr_code": [000, "top_line"],  # If not found
            "game_subtype": [000, "bottom_line"],  # from bottom to top
            "cards": ""
        },
        # "lotto_regular": "",
        # "lotto_strong": "",
        # "lotto_systematic": ""
    }

    def __init__(self, img_path):
        self.img_path = img_path
        self.filename = os.path.basename(img_path).split('.')[0]
        self.original_img = cv2.imread(self.img_path)
        self.img_height, self.img_width = self.original_img.shape[:2]

        self.qr_code_info = {}
        self.qr_code_found = False

        # will like [{"min_x": min_x, "max_x": max_x, "y": y}, {"min_x": min_x, "max_x": max_x, "y": y}]
        self.two_longest_lines = []
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
            'table': [
                [],
                [],
                []
            ]

        }
        contrast = 1.1  # Contrast control ( 0 to 127)
        brightness = .1  # Brightness control (0-100)
        contrasted_img = cv2.addWeighted(self.original_img, contrast, self.original_img, 0, brightness)
        blured_img = cv2.GaussianBlur(contrasted_img, [5, 5], 0)
        grayImage = cv2.cvtColor(blured_img, cv2.COLOR_BGR2GRAY)
        thresh, self.wb_blured_img = cv2.threshold(grayImage, 170, 255, cv2.THRESH_BINARY)
        self._save_debug_img(self.wb_blured_img, f"wb_blured_img/wb_blured_img({self.filename}).jpg")

        contrast = 1.01  # Contrast control ( 0 to 127)
        brightness = 1  # Brightness control (0-100)
        contrasted_img = cv2.addWeighted(self.original_img, contrast, self.original_img, 0, brightness)
        grayImage = cv2.cvtColor(contrasted_img, cv2.COLOR_BGR2GRAY)
        _, self.wb_img = cv2.threshold(grayImage, 100, 255, cv2.THRESH_BINARY)
        self._save_debug_img(self.wb_img, f"wb_img/wb_img({self.filename}).jpg")
        # cv2.imshow('', self.wb_img)
        # cv2.waitKey(0)
        # exit()

    @staticmethod
    def _save_debug_img(img, filepath):
        if not os.path.exists("debug_img"):
            os.mkdir("debug_img")

        splited_path = os.path.split(filepath)
        if splited_path[0] != '':
            folder = splited_path[0]
            if not os.path.exists(os.path.join("debug_img", folder)):
                os.mkdir(os.path.join("debug_img", folder))

        debug_path = os.path.join("debug_img", filepath)
        cv2.imwrite(debug_path, img)

    def get_qr_code_auto(self):
        detect = cv2.QRCodeDetector()
        qreader = QReader()
        link, corners_coord, _ = detect.detectAndDecode(self.wb_img)

        if not link:
            link, corners_coord, _ = detect.detectAndDecode(self.original_img)

        if link:
            corners_coord = corners_coord[0].tolist()
        else:
            # Затратно по времени, но очень редко вызывается
            decoded_text = qreader.detect_and_decode(image=self.wb_img, return_detections=True)
            link = decoded_text[0][0]
            corners_coord = decoded_text[1][0]["bbox_xyxy"].tolist()

            corners_coord = [
                [corners_coord[0], corners_coord[1]],
                [corners_coord[2], corners_coord[1]],
                [corners_coord[2], corners_coord[3]],
                [corners_coord[0], corners_coord[3]]]

        if link:
            self.qr_code_info = {
                'link': link,
                'top_left': (int(corners_coord[0][0]), int(corners_coord[0][1])),
                'top_right': (int(corners_coord[1][0]), int(corners_coord[1][1])),
                'bottom_right': (int(corners_coord[2][0]), int(corners_coord[2][1])),
                'bottom_left': (int(corners_coord[3][0]), int(corners_coord[3][1])),
                'middle_line': int(corners_coord[1][0] - corners_coord[0][0])  # value by OX
            }
            self.qr_code_found = True

    def get_coords_of_main_lines(self):
        gray = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2GRAY)
        img = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        detect_horizontal = cv2.morphologyEx(img, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        contours_points = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # One of elements it is data what we do not really need
        contours_points = contours_points[0] if len(contours_points) == 2 else contours_points[1]
        contours_points = list(contours_points)

        # contours_points - list of numpy arrays with elements: [[[x, y]], [[x, y]], [[x, y]]]

        # Filter None values
        sorted_contours_points = filter(lambda line_array: line_array is not None, contours_points)

        # Numpy array to python list
        lines_x_y_points = list(map(lambda line_array: line_array.tolist(), sorted_contours_points))

        # Removing unwanted nesting
        # From [[[x, y]], [[x, y]], [[x, y]]] to [[x, y], [x, y], [x, y]]  (One element it is points for one line)
        result_lines_points = list(
            map(lambda list_of_x_y_pairs: list(
                map(lambda x_y_pair: x_y_pair[0], list_of_x_y_pairs)
            ), lines_x_y_points)
        )

        # We need only two lines around cards/table
        # So this two lines have the most points count(width)
        line_threshold = self.img_width//5 * 3  # if line width >= threshold then we take this line

        for line_points in result_lines_points:
            min_x, max_x, sum_y = line_points[0][0], 0, 0
            for x, y in line_points:
                sum_y += y
                if min_x > x:
                    min_x = x
                if max_x < x:
                    max_x = x
            y = sum_y // len(line_points)

            # Remove lines which OY coord in range from 0 to 1200 and from img.height to img.height-500
            # Because card/table lines not in those diapason
            # and if x2 - x1 >= threshold

            condition = max_x - min_x >= line_threshold and y not in range(0, 1200) and y not in range(self.img_height-500, self.img_height)
            if condition:
                self.two_longest_lines.append({"min_x": min_x, "max_x": max_x, "y": y})
        self.two_longest_lines.sort(key=lambda line_data: line_data["y"])
        if len(self.two_longest_lines) == 2:
            pass
        elif len(self.two_longest_lines) == 0:
            print(Style.BRIGHT + Back.RED + Fore.WHITE + "LINES NOT FOUND")
        elif len(self.two_longest_lines) == 1:
            print(Style.BRIGHT + Back.RED + Fore.WHITE + "NOT ALL LINES WERE FOUND")
        else:
            # Here in self.two_longest_lines can be duplicates of lines with +- same OY coord
            unique_lines = []
            for index, line_dict in enumerate(self.two_longest_lines):
                if index == 0:
                    unique_lines.append(line_dict)
                    continue
                else:
                    # Add line in two lines result if OY of current line bigger than 10 than previous line
                    last_unique_line = unique_lines[-1]
                    if line_dict["y"] - last_unique_line["y"] >= 10:
                        unique_lines.append(line_dict)
            self.two_longest_lines = unique_lines

            if len(self.two_longest_lines) != 2:
                raise Exception("Error. there are not two lines")

    def get_game_type(self):
        top_line = self.two_longest_lines[0]
        crop_img = self.wb_blured_img[top_line["y"] - 190 - 300:top_line["y"] - 190,
                   self.qr_code_info['top_left'][0] - 158:self.qr_code_info['top_right'][0] + 158]

        parsed_type = self.get_value_from_image(crop_img, 'title')
        return parsed_type

    def get_game_subtype(self):
        bottom_line = self.two_longest_lines[1]
        crop_img = self.wb_blured_img[bottom_line["y"] - 75:bottom_line["y"],
                   self.qr_code_info['top_left'][0] - 158:self.qr_code_info['top_right'][0] + 158]
        # parsed_subtype = self.get_value_from_image(crop_img, 'subtype')
        # return

    def get_spent_money(self):
        bottom_line = self.two_longest_lines[1]
        crop_img = self.wb_blured_img[bottom_line["y"] + 240:bottom_line["y"] + 240 + 40,
                            self.qr_code_info['top_left'][0] - 125:self.qr_code_info['top_left'][0] + 75]
        self._save_debug_img(crop_img, f"cropped_spent_money({self.filename}).jpg")
        parsed_numbers = self.get_value_from_image(crop_img, 'numbers', parse_just_in_numbers=True)

        numbers = ''
        for symbol in parsed_numbers:
            if symbol.isnumeric():
                numbers += symbol
        return int(numbers[0:len(numbers)-2])  # TODO: int or float ???

    def get_check_dashed_number(self):
        crop_img = self.wb_blured_img[self.img_height - 75:self.img_height,
                            self.qr_code_info['top_left'][0] - 50:self.qr_code_info['top_right'][0] + 50]
        self._save_debug_img(crop_img, f"cropped_dashed_number({self.filename}).jpg")
        numbers = self.get_value_from_image(crop_img, 'numbers', parse_just_in_numbers=True)

        if len(numbers) == 19:
            dashed_number = f"{numbers[:4]}-{numbers[4:13]}-{numbers[13:]}"
            return dashed_number
        else:
            raise Exception(f"The length of the number is not correct(dashed_number)\nNumber: {numbers}")

    def get_check_spaced_number(self):
        bottom_line = self.two_longest_lines[1]
        crop_img = self.wb_blured_img[bottom_line["y"] + 340:bottom_line["y"] + 340 + 40,
                            self.qr_code_info['top_left'][0] - 125:self.qr_code_info['top_right'][0] + 125]
        self._save_debug_img(crop_img, f"cropped_spaced_number({self.filename}).jpg")
        numbers = self.get_value_from_image(crop_img, 'numbers')
        if len(numbers) == 26:
            spaced_number = f"{numbers[:8]} {numbers[8:17]} {numbers[17:]}"
            return spaced_number
        else:
            raise Exception(f"The length of the number is not correct(spaced_number)\nNumber: {numbers}")

    def get_cards(self):
        crop_img = self.wb_blured_img[self.two_longest_lines[0]["y"]:self.two_longest_lines[1]["y"],
                            self.qr_code_info['top_left'][0] - 200:self.qr_code_info['top_right'][0] + 200]
        self._save_debug_img(crop_img, f"cropped_cards({self.filename}).jpg")
        cards = self.get_value_from_image(crop_img, 'cards')
        if len(cards.split()) <= 4:
            return cards
        else:
            raise Exception(f"The count of the cards is not correct(cards)\nCards: {cards}")

    def get_table(self):
        ...

    def get_date(self):
        bottom_line = self.two_longest_lines[1]
        crop_img = self.wb_blured_img[bottom_line["y"] + 60:bottom_line["y"] + 60 + 30,
                            self.qr_code_info['top_left'][0] - 85:self.qr_code_info['top_left'][0] + 245]
        self._save_debug_img(crop_img, f"cropped_date({self.filename}).jpg")
        numbers = self.get_value_from_image(crop_img, 'numbers', parse_just_in_numbers=True)
        if len(numbers) == 12:
            return f"{numbers[:2]}:{numbers[2:4]}:{numbers[4:6]} {numbers[6:8]}.{numbers[8:10]}.{numbers[10:]}"
        else:
            raise Exception(f"The length of the number is not correct(date)\nNumber: {numbers}")

    def get_game_id(self):  # TODO: Parse brackets too
        bottom_line = self.two_longest_lines[1]
        crop_img = self.wb_blured_img[bottom_line["y"] + 20:bottom_line["y"] + 20 + 40,
                            self.qr_code_info['top_left'][0] - 120:self.qr_code_info['top_left'][0] + 90]
        self._save_debug_img(crop_img, f"cropped_game_id({self.filename}).jpg")
        numbers = self.get_value_from_image(crop_img, 'numbers', parse_just_in_numbers=True)
        return numbers

    def get_value_from_image(self, cropped_img, data_type, parse_just_in_numbers=False):
        min_width_and_height = {  # For cropped elements
            "numbers": [5, 11],
            "cards": [100, 150],
            "title": [50, 70],
            "date": [5, 11]
        }
        edges = cv2.Canny(cropped_img, 50, 200)

        contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Переводим границы в кортеж
        info_of_contours = [cv2.boundingRect(contour) for contour in contours]  # (x, y, w, h)
        # Выстраиваем элементы в правильном порядке
        sorted_contours = sorted(info_of_contours, key=lambda contour: contour[0])
        # Убираем мусор
        sorted_contours = [contour for contour in sorted_contours if contour[2] >= min_width_and_height[data_type][0] and contour[3] >= min_width_and_height[data_type][1]]

        # Убираем дубликаты элементов(если такие есть)
        unique_contours = []
        for index, contour in enumerate(sorted_contours):
            if index == 0:
                unique_contours.append(contour)
                continue
            else:
                # Если координата "х" следующего контура больше минимум на 13, то добавляем его в конечный список
                last_unique_contour = unique_contours[-1]
                if contour[0] - last_unique_contour[0] >= 13:
                    unique_contours.append(contour)
                else:
                    # Иначе проверяем площадь, если она больше у текущего элемента, то меняем на него
                    last_unique_contour_square = last_unique_contour[2] * last_unique_contour[3]
                    contour_square = contour[2] * contour[3]
                    if contour_square > last_unique_contour_square:
                        unique_contours.pop()
                        unique_contours.append(contour)

        result = ''
        if data_type == "title":
            img = cv2.imread('game_types.png')
            first_contour = unique_contours[0]
            last_contour = unique_contours[-1]
            check_title = cropped_img[last_contour[1]: first_contour[1]+first_contour[3],
                          first_contour[0]:last_contour[0]+last_contour[2]]
            self._save_debug_img(check_title, f"cropped_game_type({self.filename}).jpg")
            res = cv2.matchTemplate(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cv2.cvtColor(check_title, cv2.COLOR_BGR2RGB), cv2.TM_CCOEFF_NORMED)
            y, x = np.unravel_index(res.argmax(), res.shape)
            for key, value in game_types.items():
                if x in range(*value[0]) and y in range(*value[1]):
                    result = key
                    break
            return result

        for contour in unique_contours:
            x, y, w, h = contour
            # print(x, y, w, h)
            if data_type == "cards":
                cropped_contour = cropped_img[y-2:y + h, x-2:x + w]
            else:
                cropped_contour = cropped_img[y:y + h, x:x + w]

            if data_type == 'numbers':
                if parse_just_in_numbers:
                    img = cv2.imread('numbers.png')
                else:
                    img = cv2.imread('numbers_and_letters.png')
                    # print(cropped_contour)

                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                cropped_contour = cv2.copyMakeBorder(
                    cropped_contour,
                    top=2,
                    bottom=2,
                    left=2,
                    right=2,
                    borderType=cv2.BORDER_CONSTANT,
                    value=[255, 255, 255]
                )
                if cropped_contour is not None:
                    res = cv2.matchTemplate(img, cropped_contour, cv2.TM_CCOEFF_NORMED)
                    y, x = np.unravel_index(res.argmax(), res.shape)

                    for key, value in d_all_symbols.items():
                        if x in range(*value[0]) and y in range(*value[1]):
                            result += key

            elif data_type == 'table':
                pass

            elif data_type == 'cards':
                img = cv2.imread('cards.png')
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                cropped_contour = cv2.copyMakeBorder(
                    cropped_contour,
                    top=2,
                    bottom=2,
                    left=2,
                    right=2,
                    borderType=cv2.BORDER_CONSTANT,
                    value=[255, 255, 255]
                )

                res = cv2.matchTemplate(img, cropped_contour, cv2.TM_CCOEFF_NORMED)
                y, x = np.unravel_index(res.argmax(), res.shape)
                for key, value in cards.items():
                    if x in range(*value[0]) and y in range(*value[1]):
                        result += key + " "
        return result

    def get_result(self):
        self.get_coords_of_main_lines()
        # print("Content lines:", self.two_longest_lines)  # TODO: test print
        self.get_qr_code_auto()

        if self.qr_code_found:
            print("QR code link:", self.qr_code_info["link"])
            print("Spaced number:", self.get_check_spaced_number())
            print("Dashed number:", self.get_check_dashed_number())
            print("Spent money:", self.get_spent_money())
            print("Date:", self.get_date())
            print("Game_id:", self.get_game_id())
            game_type = self.get_game_type()
            print("Game type:", game_type)
            print("Game subtype:", self.get_game_subtype())

            if game_type == "chance":
                print("Cards:", self.get_cards())
            else:
                print("Table:", self.get_table())
        else:
            # TODO: Find qr code by hands
            print(Style.BRIGHT + Back.RED + Fore.WHITE + "QR CODE WAS NOT FOUND")

# TODO: Нужны билеты lotto(all types), 123, 777 col8, col9


# Code for testing
checks_count = 0
check_files = os.listdir("images/cards_game")
for file in check_files:
    path = f"images/cards_game/{file}"
    if os.path.isfile(path):
        print(Style.BRIGHT + Back.WHITE + Fore.BLUE + "**********************************************")
        print("Current check:", path)
        info = GetCheckGameType(path).get_result()
        print(Style.BRIGHT + Back.WHITE + Fore.BLUE + "**********************************************")
        print("")
        checks_count += 1
print(Style.BRIGHT + Back.GREEN + Fore.WHITE + f"Count of checks: {checks_count}")
