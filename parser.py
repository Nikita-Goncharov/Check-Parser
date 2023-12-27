import os
import logging
import datetime

import cv2
import numpy as np
from qreader import QReader
from colorama import init, Fore, Back, Style
from elements_coords_in_img import game_types, d_numbers, d_all_symbols, d_table_123_numbers, d_table_lotto_numbers, d_table_777_numbers, cards

init(autoreset=True)
logging.basicConfig(level=logging.INFO, filename="parser_log.log", filemode="w", style="$")


# Content lines in check it is elements relative to which we are looking for the remaining elements
class CheckParser:
    QR_CODE_RANGE = range(0, 1000)  # Diapason of coords by OX and OY
    MAX_VALID_IMG_WIDTH = 1000

    # Distance from lines to needed elements
    games_elements_distance = {
        "123": {
            "spaced_number": [340, "bottom_line"],
            "date": [55, "bottom_line"],
            "game_id": [20, "bottom_line"],
            "sum": [240, "bottom_line"],
            "game_subtype": [75, "bottom_line"],
            "game_type": [180, "top_line"],
        },
        "777": {
            "spaced_number": [310, "bottom_line"],
            "date": [55, "bottom_line"],
            "game_id": [20, "bottom_line"],
            "sum": [210, "bottom_line"],
            "game_subtype": [75, "bottom_line"],
            "game_type": [180, "top_line"],
        },
        "chance": {
            "spaced_number": [340, "bottom_line"],
            "date": [55, "bottom_line"],
            "game_id": [20, "bottom_line"],
            "sum": [240, "bottom_line"],
            "game_subtype": [75, "bottom_line"],
            "game_type": [190, "top_line"],
        },

        "lotto": {
            "spaced_number": [310, "bottom_line"],
            "date": [55, "bottom_line"],
            "game_id": [20, "bottom_line"],
            "sum": [210, "bottom_line"],
            "game_subtype": [75, "middle_line"],
            "game_type": [000, "top_line"],
        }
    }

    def __init__(self, img_path):
        self.img_path = img_path
        self.filename = os.path.basename(img_path).split(".")[0]
        self.original_img = cv2.imread(self.img_path)
        self.original_img_grayscale = cv2.imread(self.img_path, cv2.IMREAD_GRAYSCALE)
        self.img_height, self.img_width = self.original_img.shape[:2]
        self.qr_code_info = {}
        self.qr_code_found = False

        # will like {
        # "top_border_line": {...},
        # "top_line": {"min_x": min_x, "max_x": max_x, "y": y},
        # "middle_line": {...},
        # "bottom_line": {"min_x": min_x, "max_x": max_x, "y": y}
        # }
        self.longest_lines = {}
        self.bottom_oy_border_for_table = 0
        self.check_info = {
            "qr_code_link": "",
            "date": "",
            "game_type": "",
            "game_subtype": "",
            "game_id": "",
            "spent_on_ticket": 0.0,
            "dashed_number": "",
            "spaced_number": "",
            "extra": False,
            "extra_number": "",
            "cards": {
                "hearts": [],
                "diamonds": [],
                "spades": [],
                "clubs": []
            },
            "table": [

            ]
        }

        contrast = .99  # (0-127)
        brightness = .1  # (0-100)
        contrasted_img = cv2.addWeighted(
            self.original_img_grayscale,
            contrast,
            self.original_img_grayscale, 0,
            brightness
        )
        blured_img_main = cv2.bilateralFilter(contrasted_img, 1, 75, 75)

        im_bw = cv2.threshold(blured_img_main, 164, 255, cv2.THRESH_TOZERO)[1]
        self.wb_blured_img = cv2.threshold(im_bw, 130, 255, cv2.THRESH_BINARY)[1]
        self._save_debug_img(self.wb_blured_img, f"wb_blured/wb_blured({self.filename}).jpg")

    def get_game_type(self):
        crop_img = self.wb_blured_img[
                   self.qr_code_info["bottom_left"][1] + 310:self.qr_code_info["bottom_left"][1] + 310 + 300,
                   self.qr_code_info["top_left"][0] - 158:self.qr_code_info["top_right"][0] + 158
        ]
        self._save_debug_img(crop_img, f"game_type/game_type_not_cropped({self.filename}).jpg")
        parsed_type = self.get_value_from_image(crop_img, "game_type")
        self.check_info["game_type"] = parsed_type
        return parsed_type

    def _rotate_img(self):
        self.original_img = cv2.rotate(self.original_img, cv2.ROTATE_90_CLOCKWISE)
        self.wb_blured_img = cv2.rotate(self.wb_blured_img, cv2.ROTATE_90_CLOCKWISE)

        self.img_width, self.img_height = self.img_height, self.img_width

        self._save_debug_img(self.wb_blured_img, f"wb_blured/wb_blured({self.filename}).jpg")

    def _is_valid_img(self):
        response = self.try_get_qr_code()
        response_status = response["success"]
        if response_status:
            qr_corners_coords = response["corners_coords"]
            top_left_OY = int(qr_corners_coords[0][1])
            top_left_OX = int(qr_corners_coords[0][0])
            if top_left_OY not in self.QR_CODE_RANGE:  # Vertical(img) QR code at bottom
                self._rotate_img()
                self._rotate_img()
                self.try_get_qr_code()  # Update qr code coords
            elif self.img_width > self.MAX_VALID_IMG_WIDTH:  # Horizontal(img)
                if top_left_OX not in self.QR_CODE_RANGE:  # QR code at right
                    self._rotate_img()
                    self._rotate_img()
                    self._rotate_img()
                    self.try_get_qr_code()  # Update qr code coords
                else:  # QR code at left
                    self._rotate_img()
                    self.try_get_qr_code()  # Update qr code coords
            return {"success": True, "description": ""}
        else:
            return {"success": False, "description": response["description"]}

    @staticmethod
    def _save_debug_img(img, filepath):
        if not os.path.exists("debug_img"):
            os.mkdir("debug_img")

        splited_path = os.path.split(filepath)
        if splited_path[0] != "":
            folder = splited_path[0]
            if not os.path.exists(os.path.join("debug_img", folder)):
                os.mkdir(os.path.join("debug_img", folder))

        debug_path = os.path.join("debug_img", filepath)
        cv2.imwrite(debug_path, img)

    def try_get_qr_code(self):
        detect = cv2.QRCodeDetector()  # qrcode
        qreader = QReader()
        link, corners_coords, _ = detect.detectAndDecode(self.wb_blured_img)

        if not link:
            link, corners_coords, _ = detect.detectAndDecode(self.original_img)

        if link:
            corners_coords = corners_coords[0].tolist()
        else:
            # Затратно по времени, но очень редко вызывается
            decoded_text = qreader.detect_and_decode(image=self.wb_blured_img, return_detections=True)
            link = decoded_text[0][0]
            corners_coords = decoded_text[1][0]["bbox_xyxy"].tolist()

            corners_coords = [
                [corners_coords[0], corners_coords[1]],
                [corners_coords[2], corners_coords[1]],
                [corners_coords[2], corners_coords[3]],
                [corners_coords[0], corners_coords[3]]]

        if link:
            self._save_qr_code(link, corners_coords)
            return {"success": True, "description": "", "corners_coords": corners_coords}
        else:
            return {"success": False, "description": "Qr code is not detected"}

    def _save_qr_code(self, link, corners_coords):
        self.qr_code_info = {
            "link": link,
            "top_left": (int(corners_coords[0][0]), int(corners_coords[0][1])),
            "top_right": (int(corners_coords[1][0]), int(corners_coords[1][1])),
            "bottom_right": (int(corners_coords[2][0]), int(corners_coords[2][1])),
            "bottom_left": (int(corners_coords[3][0]), int(corners_coords[3][1])),
            "middle_line": int(corners_coords[1][0] - corners_coords[0][0])  # value by OX
        }
        self.qr_code_found = True

        self.check_info["qr_code_link"] = link

    def get_coords_of_main_lines(self):
        blured_img_lines = cv2.GaussianBlur(self.original_img_grayscale, [5, 5], 0)
        img = cv2.threshold(blured_img_lines, 180, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        self._save_debug_img(img, f"wb_inverted/wb_inverted({self.filename}).jpg")

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        img_dilation = cv2.dilate(img, kernel, iterations=2)
        detect_horizontal = cv2.morphologyEx(img_dilation, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

        contours_points = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # One of elements it is data what we do not really need
        # contours_points - list of numpy arrays with elements: [[[x, y]], [[x, y]], [[x, y]]]
        contours_points = contours_points[0] if len(contours_points) == 2 else contours_points[1]
        contours_points = list(contours_points)
        logging.info(f"Contour points: {contours_points}")

        # Filter None values
        sorted_contours_points = filter(lambda line_array: line_array is not None, contours_points)

        # Numpy array to python list
        lines_x_y_points = list(map(lambda line_array: line_array.tolist(), sorted_contours_points))
        logging.info(f"Numpy arrays to python list: {lines_x_y_points}")

        # Removing unwanted nesting
        # From [[[x, y]], [[x, y]], [[x, y]]] to [[x, y], [x, y], [x, y]]  (One element it is points for one line)
        unnested_lines_points = list(
            map(lambda list_of_x_y_pairs: list(
                map(lambda x_y_pair: x_y_pair[0], list_of_x_y_pairs)
            ), lines_x_y_points)
        )
        logging.info(f"Unnested lines points: {unnested_lines_points}")

        # Now we have lists with points of lines, but needed lines can be splited
        # So we should merge lines with +- same OY
        # unnested_lines_points already sorted from bigger to smaller
        result_lines_points = []
        for index, line_points in enumerate(unnested_lines_points):
            if index == 0:
                result_lines_points.append(line_points)
            else:
                last_line = result_lines_points[-1]
                last_point_y = last_line[-1][1]
                if last_point_y - line_points[-1][1] <= 10:
                    result_lines_points.pop()
                    result_lines_points.append([*last_line, *line_points])
                else:
                    result_lines_points.append(line_points)
        logging.info(f"Merged lines: {result_lines_points}")

        # We need only two lines around cards/table
        # So this two lines have the most points count(width)
        line_threshold = self.img_width - 100  # if line width >= threshold then we take this line

        lines = []

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
            condition = max_x - min_x >= line_threshold and y not in range(0, 1200) and y not in range(
                self.img_height - 500, self.img_height)
            if condition:
                lines.append({"min_x": min_x, "max_x": max_x, "y": y})

        if len(lines) == 0:
            print(Style.BRIGHT + Back.RED + Fore.WHITE + "LINES NOT FOUND")
        elif len(lines) == 1:
            print(Style.BRIGHT + Back.RED + Fore.WHITE + "NOT ALL LINES WERE FOUND")
        else:
            # Here in lines can be more than four lines now
            # But the four needed ones are the longest
            lines = sorted(lines, key=lambda line: line["max_x"] - line["min_x"], reverse=True)
            lines = lines[:4]
            lines.sort(key=lambda line_data: line_data["y"])
            # print("Sorted lines:", lines)
            if len(lines) == 2:  # 123, 777, chance
                self.longest_lines = {
                    "top_border_line": {},
                    "top_line": lines[0],
                    "middle_line": {},
                    "bottom_line": lines[1]
                }
            elif len(lines) == 3:  # For lotto, if three main lines
                self.longest_lines = {
                    "top_border_line": {},
                    "top_line": lines[0],
                    "middle_line": lines[1],
                    "bottom_line": lines[2]
                }
            else:  # For lotto, if four main lines
                self.longest_lines = {
                    "top_border_line": lines[0],
                    "top_line": lines[1],
                    "middle_line": lines[2],
                    "bottom_line": lines[3]
                }

    def get_game_subtype(self):
        img = cv2.imread("no_needed_data.png")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        game_type = self.check_info["game_type"]
        distance_line = self.games_elements_distance[game_type]["game_subtype"]
        distance, line = distance_line[0], self.longest_lines[distance_line[1]]

        crop_img = self.wb_blured_img[
            line["y"] - distance:line["y"],
            self.qr_code_info["top_left"][0] - 158:self.qr_code_info["top_right"][0] + 158
        ]

        res = cv2.matchTemplate(crop_img, img, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= 0.7)  # THRESHOLD
        if len(loc[0].tolist()) != 0:
            crop_img = self.wb_blured_img[
                line["y"] - 115:line["y"] - 50,
                self.qr_code_info["top_left"][0] - 158:self.qr_code_info["top_right"][0] + 158
            ]
            self.bottom_oy_border_for_table = self.longest_lines["bottom_line"]["y"] - 55

        self._save_debug_img(crop_img, f"subtype/subtype({self.filename}).jpg")
        parsed_subtype = self.get_value_from_image(crop_img, "game_subtype")
        self.check_info["game_subtype"] = parsed_subtype
        return parsed_subtype

    def get_spent_money(self):
        game_type = self.check_info["game_type"]
        distance_line = self.games_elements_distance[game_type]["sum"]
        distance, line = distance_line[0], self.longest_lines[distance_line[1]]

        crop_img = self.wb_blured_img[
            line["y"] + distance:line["y"] + distance + 40,
            self.qr_code_info["top_left"][0] - 125:self.qr_code_info["top_left"][0] + 90
        ]
        self._save_debug_img(crop_img, f"spent_money/spent_money({self.filename}).jpg")
        parsed_numbers = self.get_value_from_image(crop_img, "sum")

        numbers = ""
        for symbol in parsed_numbers:
            if symbol.isnumeric():
                numbers += symbol
        numbers = numbers[0:-2] + "." + numbers[-2:]
        self.check_info["spent_on_ticket"] = float(numbers[0:len(numbers)])
        return self.check_info["spent_on_ticket"]

    def get_check_dashed_number(self):
        crop_img = self.wb_blured_img[
            self.img_height - 75:self.img_height - 30,
            self.qr_code_info["top_left"][0] - 55:self.qr_code_info["top_right"][0] + 55
        ]
        self._save_debug_img(crop_img, f"dashed_number/dashed_number({self.filename}).jpg")
        numbers = self.get_value_from_image(crop_img, "numbers")

        if len(numbers) == 19:
            dashed_number = f"{numbers[:4]}-{numbers[4:13]}-{numbers[13:]}"
            self.check_info["dashed_number"] = dashed_number
            return dashed_number
        else:
            print(numbers)
            cv2.imshow('', crop_img)
            cv2.waitKey(0)
            raise Exception(f"The length of the number is not correct(dashed_number)\nNumber: {numbers}")

    def get_check_spaced_number(self):
        game_type = self.check_info["game_type"]
        distance_line = self.games_elements_distance[game_type]["spaced_number"]
        distance, line = distance_line[0], self.longest_lines[distance_line[1]]

        crop_img = self.wb_blured_img[
            line["y"] + distance:line["y"] + distance + 40,
            self.qr_code_info["top_left"][0] - 130:self.qr_code_info["top_right"][0] + 130
        ]
        self._save_debug_img(crop_img, f"spaced_number/spaced_number({self.filename}).jpg")
        numbers = self.get_value_from_image(crop_img, "numbers", parse_just_in_numbers=False)
        if len(numbers) == 26:
            spaced_number = f"{numbers[:8]} {numbers[8:17]} {numbers[17:]}"
            self.check_info["spaced_number"] = spaced_number
            return spaced_number
        else:
            print(numbers)
            cv2.imshow('', crop_img)
            cv2.waitKey(0)
            raise Exception(f"The length of the number is not correct(spaced_number)\nNumber: {numbers}")

    def get_cards(self):  # TODO: split cards ??? transfer one at a time to get_value_from_image
        top_line = self.longest_lines["top_line"]
        bottom_line = self.longest_lines["bottom_line"]

        crop_img = self.wb_blured_img[
            top_line["y"]:bottom_line["y"],
            self.qr_code_info["top_left"][0] - 200:self.qr_code_info["top_right"][0] + 200
        ]
        self._save_debug_img(crop_img, f"cards/cards({self.filename}).jpg")
        self.get_value_from_image(crop_img, "cards")

    def get_table_123_777(self):  # TODO: make checking for duplicates of numbers
        game_type = self.check_info["game_type"]
        if self.bottom_oy_border_for_table == 0:
            bottom_oy_border = self.longest_lines["bottom_line"]["y"]
        else:
            # finding from top_line to top of subtype if exists
            bottom_oy_border = self.bottom_oy_border_for_table

        crop_img = self.wb_blured_img[
            self.longest_lines["top_line"]["y"]:bottom_oy_border,
            0:self.img_width
        ]
        self._save_debug_img(crop_img, f"table/table({self.filename}).jpg")

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        img_dilated = cv2.dilate(cv2.threshold(crop_img, 127, 255, cv2.THRESH_BINARY_INV)[1], kernel)

        contours = cv2.findContours(img_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        info_of_contours = [cv2.boundingRect(contour) for contour in contours]  # (x, y, w, h)

        # Remove garbage contours
        min_width = 35 if game_type == "123" else 17
        max_width = 50 if game_type == "123" else 35
        min_height = 35 if game_type == "123" else 35


        sorted_contours = [contour for contour in info_of_contours if
                           contour[2] >= min_width and contour[2] <= max_width and contour[3] >= min_height]

        if game_type == "123":
            if len(sorted_contours) % 3 != 0:
                raise Exception("Not all numbers were found(table_123)")

            # sort contours in right way n11, n21, n31 n12, n22, n32
            sorted_by_OY = sorted(sorted_contours, key=lambda contour: contour[1])
            # group numbers by 3 and sort by OX
            groups_of_numbers = []
            i = 0
            while len(groups_of_numbers) != len(sorted_by_OY) / 3:
                next_three_numbers = sorted_by_OY[i:i+3]
                next_three_numbers.sort(key=lambda contour: contour[0])  # sort by OX
                groups_of_numbers.append(next_three_numbers)
                i += 3

            for index, group in enumerate(groups_of_numbers):
                self.check_info["table"].append([])
                for number in group:
                    crop_number = crop_img[
                        number[1]:number[1] + number[3],
                        number[0]:number[0] + number[2]
                    ]

                    table_number = self.get_value_from_image(crop_number, "table")
                    self.check_info["table"][index].append(int(table_number))
        elif game_type == "777":
            current_line = 0
            numbers_in_lines = []
            prev_number_OY = 0
            sorted_by_OY = sorted(sorted_contours, key=lambda contour: contour[1])
            numbers_in_lines.append([])

            for number in sorted_by_OY:
                if prev_number_OY == 0:
                    prev_number_OY = number[1]

                if number[1] - prev_number_OY > 10:
                    current_line += 1
                    numbers_in_lines.append([])

                numbers_in_lines[current_line].append(number)

                prev_number_OY = number[1]

            for index, line in enumerate(numbers_in_lines):
                line.sort(key=lambda number: number[0])
                numbers_in_lines[index] = line

            for index, line in enumerate(numbers_in_lines):
                self.check_info["table"].append([])
                for number in line:
                    crop_number = crop_img[
                        number[1]:number[1] + number[3],
                        number[0]:number[0] + number[2]
                    ]
                    table_number = self.get_value_from_image(crop_number, "table")
                    self.check_info["table"][index].append(table_number)

            # merge 777 numbers by two
            print(self.check_info["table"])
            for index, line in enumerate(self.check_info["table"]):
                new_line = []
                i = 0
                while i < len(line):
                    first_number = line[i]
                    second_number = line[i+1]
                    new_line.append(int(first_number+second_number))
                    i += 2
                self.check_info["table"][index] = new_line

        return self.check_info["table"]

    def get_table_lotto(self):  # TODO: find lotto table
        # if self.bottom_oy_border_for_table == 0:
        #     bottom_oy_border = self.longest_lines["middle_line"]["y"]
        # else:
        #     # finding from top_line to top of subtype if exists
        #     bottom_oy_border = self.bottom_oy_border_for_table

        bottom_oy_border = self.longest_lines["middle_line"]["y"]

        crop_img = self.wb_blured_img[
            self.longest_lines["top_line"]["y"]:bottom_oy_border,
            0:self.img_width
        ]
        self._save_debug_img(crop_img, f"table/table({self.filename}).jpg")
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        img_dilated = cv2.dilate(cv2.threshold(crop_img, 127, 255, cv2.THRESH_BINARY_INV)[1], kernel)

        # table_number = self.get_value_from_image(crop_number, "table")

        # cv2.imshow('', img_dilated)
        # cv2.waitKey(0)

    def get_extra(self):
        middle_line = self.longest_lines["middle_line"]
        bottom_line = self.longest_lines["bottom_line"]

        crop_img = self.wb_blured_img[
            middle_line["y"]:bottom_line["y"],
            0:self.img_width
        ]
        self._save_debug_img(crop_img, f"extra/extra({self.filename}).jpg")

        w, h = crop_img.shape[:2]

        extra_crop = crop_img[
            0:130,
            self.img_width // 2:self.img_width
        ]
        self._save_debug_img(extra_crop, f"extra/is_extra({self.filename}).jpg")

        extra_number_crop = crop_img[
            140:h,
            0:self.img_width
        ]
        self._save_debug_img(extra_number_crop, f"extra/extra_number({self.filename}).jpg")

        is_extra = self.get_value_from_image(extra_crop, "is_extra")

        if is_extra:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
            img_dilated = cv2.dilate(cv2.threshold(extra_number_crop, 127, 255, cv2.THRESH_BINARY_INV)[1], kernel)

            contours = cv2.findContours(img_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            info_of_contours = [cv2.boundingRect(contour) for contour in contours]  # (x, y, w, h)
            sorted_contours = [contour for contour in info_of_contours if
                               contour[2] >= 17 and contour[2] <= 30 and contour[3] >= 25]
            sorted_by_OY = sorted(sorted_contours, key=lambda contour: contour[0])
            for number in sorted_by_OY:
                crop_number = extra_number_crop[
                    number[1]:number[1] + number[3],
                    number[0]:number[0] + number[2]
                ]
                extra_number = self.get_value_from_image(crop_number, "extra_numbers")
                self.check_info["extra_number"] += extra_number

        self.check_info["extra"] = is_extra
        return is_extra

    def get_date(self):
        game_type = self.check_info["game_type"]
        distance_line = self.games_elements_distance[game_type]["date"]
        distance, line = distance_line[0], self.longest_lines[distance_line[1]]
        crop_img = self.wb_blured_img[
            line["y"] + distance:line["y"] + distance + 30,
            self.qr_code_info["top_left"][0] - 90:self.qr_code_info["top_left"][0] + 280
        ]
        self._save_debug_img(crop_img, f"date/date({self.filename}).jpg")
        numbers = self.get_value_from_image(crop_img, "date")
        if len(numbers) == 12:
            date = f"{numbers[:2]}:{numbers[2:4]}:{numbers[4:6]} {numbers[6:8]}.{numbers[8:10]}.{numbers[10:]}"
            self.check_info["date"] = datetime.datetime.strptime(date, "%H:%M:%S %d.%m.%y")
            return date
        else:
            raise Exception(f"The length of the number is not correct(date)\nNumber: {numbers}")

    def get_game_id(self):  # TODO: find few game_ids, if exists
        game_type = self.check_info["game_type"]
        distance_line = self.games_elements_distance[game_type]["game_id"]
        distance, line = distance_line[0], self.longest_lines[distance_line[1]]
        crop_img = self.wb_blured_img[
            line["y"] + distance:line["y"] + distance + 40,
            self.qr_code_info["top_left"][0] - 120:self.qr_code_info["top_left"][0] + 90
        ]
        self._save_debug_img(crop_img, f"game_id/game_id({self.filename}).jpg")
        numbers = self.get_value_from_image(crop_img, "game_id", parse_just_in_numbers=False)
        self.check_info["game_id"] = numbers.split('(')[0]
        return numbers

    def get_value_from_image(self, cropped_img, data_type, parse_just_in_numbers=True):
        result = ""
        # data_type: numbers, date, game_id, sum, table, cards, game_type, game_subtype
        min_width_and_height = {  # For cropped contours [min_width, min_height]
            "numbers": [6, 11],
            "table": [22, 35],
            "cards": [100, 150],
            "game_type": [30, 70],
            "game_subtype": [10, 20],
            "date": [6, 11],
            "sum": [6, 11],
            "game_id": [6, 11],
            "is_extra": [40, 40],
            "extra_numbers": [22, 35]
        }

        if data_type == "sum":
            height, width = cropped_img.shape[:2]
            find_symbol_img = cv2.imread("sum_symbol_border.png")
            find_symbol_img = cv2.cvtColor(find_symbol_img, cv2.COLOR_BGR2GRAY)
            res = cv2.matchTemplate(cropped_img, find_symbol_img, cv2.TM_CCOEFF_NORMED)
            y, x = np.unravel_index(res.argmax(), res.shape)

            cropped_img = cropped_img[
                0:height,
                x+20:width
            ]

        if data_type == "table":
            game_type = self.check_info["game_type"]
            if game_type == "123":
                img = cv2.imread("table_123_numbers.png")
                d_table_numbers = d_table_123_numbers
            elif game_type == "777":
                img = cv2.imread("table_777_numbers.png")
                d_table_numbers = d_table_777_numbers
            elif game_type == "lotto":
                img = cv2.imread("table_lotto_numbers.png")
                d_table_numbers = d_table_lotto_numbers

            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            cropped_img = cv2.copyMakeBorder(
                cropped_img,
                top=1,
                bottom=1,
                left=0,
                right=0,
                borderType=cv2.BORDER_CONSTANT,
                value=[255, 255, 255]
            )
            # cv2.imshow('', cropped_img)
            # cv2.waitKey(0)
            res = cv2.matchTemplate(img, cropped_img, cv2.TM_CCOEFF_NORMED)
            y, x = np.unravel_index(res.argmax(), res.shape)
            print(x, y)
            for key, value in d_table_numbers.items():
                if x in range(*value[0]) and y in range(*value[1]):
                    # print(key)
                    return key

        if data_type == "extra_numbers":  # TODO: create img, dict for extra nums
            img = cv2.imread("table_777_numbers.png")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            cropped_img = cv2.copyMakeBorder(
                cropped_img,
                top=1,
                bottom=1,
                left=0,
                right=0,
                borderType=cv2.BORDER_CONSTANT,
                value=[255, 255, 255]
            )
            # cv2.imshow('', cropped_img)
            # cv2.waitKey(0)
            res = cv2.matchTemplate(img, cropped_img, cv2.TM_CCOEFF_NORMED)
            y, x = np.unravel_index(res.argmax(), res.shape)
            # print(x, y)
            for key, value in d_table_777_numbers.items():
                if x in range(*value[0]) and y in range(*value[1]):
                    # print(key)
                    return key

        edges = cv2.Canny(cropped_img, 10, 200)
        contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Contours to list of tuples
        info_of_contours = [cv2.boundingRect(contour) for contour in contours]  # (x, y, w, h)
        # print(info_of_contours)
        # Sort contours in right way
        sorted_contours = sorted(info_of_contours, key=lambda contour: contour[0])
        # Remove garbage contours
        sorted_contours = [contour for contour in sorted_contours if
                           contour[2] >= min_width_and_height[data_type][0] and contour[3] >=
                           min_width_and_height[data_type][1]]

        # Remove duplicates of contours
        unique_contours = []
        for index, contour in enumerate(sorted_contours):
            if index == 0:
                unique_contours.append(contour)
                continue
            else:
                # Append contour if contour OX more than prev by (contour_threshold)px
                contour_threshold = 20 if data_type == "cards" else 13
                last_unique_contour = unique_contours[-1]
                if contour[0] - last_unique_contour[0] >= contour_threshold:
                    unique_contours.append(contour)
                else:
                    # Else check square, if current bigger then swap contour in unique_contours
                    last_unique_contour_square = last_unique_contour[2] * last_unique_contour[3]
                    contour_square = contour[2] * contour[3]
                    if contour_square > last_unique_contour_square:
                        unique_contours.pop()
                        unique_contours.append(contour)

        if data_type == "game_type":
            img = cv2.imread("game_types.png")
            first_contour = unique_contours[0]
            last_contour = unique_contours[-1]
            check_title = cropped_img[
                last_contour[1]: first_contour[1] + first_contour[3],
                first_contour[0]:last_contour[0] + last_contour[2]
            ]
            self._save_debug_img(check_title, f"game_type/game_type({self.filename}).jpg")
            res = cv2.matchTemplate(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cv2.cvtColor(check_title, cv2.COLOR_BGR2RGB),
                                    cv2.TM_CCOEFF_NORMED)
            y, x = np.unravel_index(res.argmax(), res.shape)
            for key, value in game_types.items():
                if x in range(*value[0]) and y in range(*value[1]):
                    result = key
                    break
            return result

        elif data_type == "game_subtype":
            game_type = self.check_info["game_type"]
            if game_type == "777":
                img = cv2.imread("777_systematic_subtype.png")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                res = cv2.matchTemplate(cropped_img, img, cv2.TM_CCOEFF_NORMED)
                loc = np.where(res >= 0.7)  # THRESHOLD
                if len(loc[0].tolist()) == 0:
                    result = "777_regular"
                else:
                    if self.bottom_oy_border_for_table != 0:
                        self.bottom_oy_border_for_table -= 60  # if no needed data exists
                    else:
                        self.bottom_oy_border_for_table = self.longest_lines["bottom_line"]["y"] - 60
                    img = cv2.imread("777_systematic_subtype_col8.png")
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    res = cv2.matchTemplate(cropped_img, img, cv2.TM_CCOEFF_NORMED)
                    loc = np.where(res >= 0.7)  # THRESHOLD
                    if len(loc[0].tolist()) == 0:
                        result = "777_col9"
                    else:
                        result = "777_col8"

            elif game_type == "chance":
                img_type_in_subtype = cv2.imread("chance_subtype.png")
                img = cv2.cvtColor(img_type_in_subtype, cv2.COLOR_BGR2GRAY)
                res = cv2.matchTemplate(cropped_img, img, cv2.TM_CCOEFF_NORMED)
                y, x = np.unravel_index(res.argmax(), res.shape)
                # Crop exactly subtype
                cropped_subtype = cropped_img[
                    y:y+50,
                    x-180:x+101+70
                ]

                img_subtype_multi = cv2.imread("chance_multi_subtype.png")
                img_subtype_multi = cv2.cvtColor(img_subtype_multi, cv2.COLOR_BGR2GRAY)

                img_subtype_systematic = cv2.imread("chance_systematic_subtype.png")
                img_subtype_systematic = cv2.cvtColor(img_subtype_systematic, cv2.COLOR_BGR2GRAY)

                res_suffix_subtype = cv2.matchTemplate(cropped_subtype, img_subtype_multi, cv2.TM_CCOEFF_NORMED)
                loc_suffix = np.where(res_suffix_subtype >= 0.7)  # THRESHOLD

                res_prefix_subtype = cv2.matchTemplate(cropped_subtype, img_subtype_systematic, cv2.TM_CCOEFF_NORMED)
                loc_prefix = np.where(res_prefix_subtype >= 0.7)  # THRESHOLD

                if len(loc_suffix[0].tolist()) != 0:
                    result = "chance_multi"
                elif len(loc_prefix[0].tolist()) != 0:
                    result = "chance_systematic"
                else:
                    result = "chance_regular"

                if self.bottom_oy_border_for_table != 0:
                    self.bottom_oy_border_for_table -= 60  # if no needed data exists
                else:
                    self.bottom_oy_border_for_table = self.longest_lines["bottom_line"]["y"] - 60

            elif game_type == "123":
                if self.bottom_oy_border_for_table != 0:
                    self.bottom_oy_border_for_table -= 60  # if no needed data exists
                else:
                    self.bottom_oy_border_for_table = self.longest_lines["bottom_line"]["y"] - 60
                result = "123_regular"

            elif game_type == "lotto":  # TODO: subtype
                # if self.bottom_oy_border_for_table != 0:
                #     self.bottom_oy_border_for_table -= 60  # if no needed data exists
                # else:
                #     self.bottom_oy_border_for_table = self.longest_lines["bottom_line"]["y"] - 60

                # regular, strong, systematic
                pass

            return result

        elif data_type in ["numbers", "date", "sum"]:
            if not parse_just_in_numbers:
                img = cv2.imread("numbers_and_letters.png")
            else:
                img = cv2.imread("numbers.png")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            cropped_img = cv2.copyMakeBorder(
                cropped_img,
                top=3,
                bottom=3,
                left=30,
                right=10,
                borderType=cv2.BORDER_CONSTANT,
                value=[255, 255, 255]
            )

            # In sum image symbols are located far from each other(so we need make bigger kernel)
            width_of_kernel = 20 if data_type == "sum" else 15
            rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (width_of_kernel, 3))

            dilation = cv2.dilate(255-cropped_img, rect_kernel, iterations=1)

            # cv2.imshow('', dilation)
            # cv2.waitKey(0)
            contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            im2 = cropped_img.copy()  # TODO: for test
            contours = map(lambda cnt: cv2.boundingRect(cnt), contours)
            contours = sorted(contours, key=lambda contour: contour[0])
            for cnt in contours:
                x, y, w, h = cnt
                # print("Block size", x, y, w, h)
                min_block_width = 70 if data_type == "sum" else 100
                if w < min_block_width:
                    continue

                # FIXME: dashed number last part
                current_x = x + 7  # TODO: find OX of first symbol in block
                while ((x + w) - current_x) > 15:
                    cropped_contour = cropped_img[
                        y:y + h,
                        current_x:current_x + 17
                    ]

                    symbol_width, symbol_height = cropped_contour.shape[:2]
                    count_non_zero = cv2.countNonZero(cropped_contour)

                    # Find rects in which count of black pixels less than...
                    minimal_count_black_pixels = 40 if data_type == "numbers" else 60
                    if symbol_width*symbol_height - count_non_zero < minimal_count_black_pixels and parse_just_in_numbers:
                        current_x += (17 + 2)
                        continue

                    cv2.rectangle(im2, (current_x, y), (current_x + 17, y + h), (0, 0, 0), 1)
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
                    symbol_y, symbol_x = np.unravel_index(res.argmax(), res.shape)
                    print(symbol_x, symbol_y)
                    for key, value in d_all_symbols.items():
                        if symbol_x in range(*value[0]) and symbol_y in range(*value[1]):
                            result += key
                    current_x += (17 + 2)
            if len(result) != 26 and not parse_just_in_numbers or len(result) != 19 and parse_just_in_numbers and data_type == "numbers":
                cv2.imshow('', im2)
                cv2.waitKey(0)

            return result

        elif data_type == "is_extra":
            img = cv2.imread("extra_true.png")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            res = cv2.matchTemplate(cropped_img, img, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= 0.7)  # THRESHOLD
            if len(loc[0].tolist()) != 0:
                return True
            return False

        for contour in unique_contours:
            x, y, w, h = contour

            cropped_contour = cropped_img[y:y + h, x:x + w]
            cropped_contour = cv2.copyMakeBorder(
                cropped_contour,
                top=2,
                bottom=2,
                left=2,
                right=2,
                borderType=cv2.BORDER_CONSTANT,
                value=[255, 255, 255]
            )

            if data_type == "game_id":
                img = cv2.imread("numbers.png")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                if cropped_contour is not None:
                    res = cv2.matchTemplate(img, cropped_contour, cv2.TM_CCOEFF_NORMED)
                    y, x = np.unravel_index(res.argmax(), res.shape)
                    print(x, y)
                    for key, value in d_numbers.items():
                        if x in range(*value[0]) and y in range(*value[1]):
                            result += key

            elif data_type == "cards":
                img = cv2.imread("cards.png")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                res = cv2.matchTemplate(img, cropped_contour, cv2.TM_CCOEFF_NORMED)
                y, x = np.unravel_index(res.argmax(), res.shape)
                for key, value in cards.items():
                    if x in range(*value[0]) and y in range(*value[1]):
                        card, card_type = key.split("-")
                        self.check_info["cards"][card_type].append(card)
        return result

    def get_result(self):  # TODO: create raise exception in every method and do try: except:
        # All should be called in this way

        response = self._is_valid_img()
        if response["success"]:
            self.get_coords_of_main_lines()

            logging.info(f"Main check lines: {self.longest_lines}")
            logging.info(f"QR code is found: {self.qr_code_found}")

            game_type = self.get_game_type()

            logging.info(f"Game type: {game_type}")
            print("Game type:", game_type)

            print("Game subtype:", self.get_game_subtype())

            # print("QR code link:", self.qr_code_info["link"])
            print("Game_id:", self.get_game_id())
            print("Date:", self.get_date())
            print("Spent money:", self.get_spent_money())
            print("Dashed number:", self.get_check_dashed_number())
            print("Spaced number:", self.get_check_spaced_number())

            if game_type == "chance":
                self.get_cards()
                print("Cards:", self.check_info["cards"])
            elif game_type == "lotto":
                print("Table:", self.get_table_lotto())
                print("Is extra:", self.get_extra())
            else:
                print("Table:", self.get_table_123_777())
            print(self.check_info)
        else:
            print(Style.BRIGHT + Back.RED + Fore.WHITE + f"{response['description']}")

# Ask:
# TODO: in spaced number "01HVF#7R 409361001 000604494" only first part can be with alpha symbols ???

# For my self
# TODO: remove noise pixels


# Code for testing
checks_count = 0
check_folder = "lotto"  # checks directory
checks_list = os.listdir(check_folder)
logging.info(f"Checks folder: {check_folder}")

for file in checks_list:
    path = os.path.join(check_folder, file)
    logging.info(f"File path: {path}")
    if os.path.isfile(path):
        print(Style.BRIGHT + Back.WHITE + Fore.BLUE + "**********************************************")
        print("Current check:", path)
        CheckParser(path).get_result()
        print(Style.BRIGHT + Back.WHITE + Fore.BLUE + "**********************************************")
        print("")
        checks_count += 1

logging.info(f"Count of checks: {checks_count}")
print(Style.BRIGHT + Back.GREEN + Fore.WHITE + f"Count of checks: {checks_count}")
