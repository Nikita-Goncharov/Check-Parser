import os
import logging
import datetime

import cv2
import numpy as np
from qreader import QReader
from colorama import init, Fore, Back, Style
from elements_coords_in_img import game_types, d_numbers, d_all_symbols, d_table_123_numbers, d_table_lotto_numbers, \
    d_table_777_numbers, cards, d_extra_numbers

init(autoreset=True)
logging.basicConfig(level=logging.INFO, filename="parser_log.log", filemode="w", style="$")

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
        self.main_data_contours = {
            "game_ids_count": 0,
            "game_id": (),
            "date": (),
            "sum": (),
            "spaced_number": (),
            "dashed_number": ()
        }
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
            "table": {}
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

        im_bw = cv2.threshold(blured_img_main, 170, 255, cv2.THRESH_TOZERO)[1]
        self.wb_blured_img = cv2.threshold(im_bw, 10, 255, cv2.THRESH_BINARY)[1]

        denoised_img = self._denoise_wb_img(self.wb_blured_img)
        self.wb_blured_img = denoised_img

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
        self.original_img_grayscale = cv2.rotate(self.original_img_grayscale, cv2.ROTATE_90_CLOCKWISE)
        self.wb_blured_img = cv2.rotate(self.wb_blured_img, cv2.ROTATE_90_CLOCKWISE)

        self.img_width, self.img_height = self.img_height, self.img_width

        self._save_debug_img(self.wb_blured_img, f"wb_blured/wb_blured({self.filename}).jpg")

    def _is_valid_img(self):
        response = self.try_get_qr_code()
        response_status = response["success"]
        logging.info(f"Check({self.filename}) qr code status: {response_status}")
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
        debug_img_folder = "debug_img"
        if not os.path.exists(debug_img_folder):
            os.mkdir(debug_img_folder)

        splited_path = os.path.split(filepath)
        if splited_path[0] != "":
            folder = splited_path[0]
            if not os.path.exists(os.path.join(debug_img_folder, folder)):
                os.mkdir(os.path.join(debug_img_folder, folder))

        debug_path = os.path.join(debug_img_folder, filepath)
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

    @staticmethod
    def split_lotto_number_contours(number_contours, sort_by_OY=False):
        prev_number_contour = 0
        index_of_first_strong = 0
        for index, contour in enumerate(number_contours):
            if prev_number_contour != 0 and contour[0] - prev_number_contour[0] >= 100:
                index_of_first_strong = index
                break
            prev_number_contour = contour

        regular_contour_nums = number_contours[:index_of_first_strong]
        strong_contour_nums = number_contours[index_of_first_strong::]
        # because here can be numbers in few lines, so we should sort them in right way(for lotto strong and systematic)
        if sort_by_OY:
            regular_contour_nums.sort(key=lambda cnt: [1])
            strong_contour_nums.sort(key=lambda cnt: [1])
        return regular_contour_nums, strong_contour_nums

    @staticmethod
    def _denoise_wb_img(cropped_img, noise_width=5, noise_height=5):
        contours = cv2.findContours(cropped_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]  # CHAIN_APPROX_NONE
        contours = [cv2.boundingRect(cnt) for cnt in contours]

        contours = sorted(contours, key=lambda cnt: cnt[2] * cnt[3])
        for cnt in contours:
            x, y, w, h = cnt
            if w < noise_width or h < noise_height:
                for i in range(h):
                    for j in range(w):
                        cropped_img[y + i, x + j] = 255
            else:
                break
        return cropped_img

    def get_coords_of_main_lines(self):
        blured_img_lines = cv2.GaussianBlur(self.original_img_grayscale, [5, 5], 0)
        img = cv2.threshold(blured_img_lines, 180, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        self._save_debug_img(img, f"wb_inverted/wb_inverted({self.filename}).jpg")

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        img_dilation = cv2.dilate(img, kernel, iterations=2)
        detect_horizontal = cv2.morphologyEx(img_dilation, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        self._save_debug_img(detect_horizontal, f"detect_lines/detect_lines({self.filename}).jpg")

        contours_points = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # One of elements it is data what we do not really need
        # contours_points - list of numpy arrays with elements: [[[x, y]], [[x, y]], [[x, y]]]
        contours_points = contours_points[0] if len(contours_points) == 2 else contours_points[1]
        contours_points = list(contours_points)

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
            print(self.longest_lines)

    def get_main_data_contours(self):
        def split_contours_by_width(contours):
            index_of_first_small_contour = 0
            for index, contour in enumerate(contours):
                if contour[2] < 400:
                    index_of_first_small_contour = index
                    break

            large_contours = contours[:index_of_first_small_contour]
            small_contours = contours[index_of_first_small_contour::]

            return large_contours, small_contours

        bottom_line = self.longest_lines["bottom_line"]
        initial_img = self.wb_blured_img[
            bottom_line["y"]:self.img_height,
            0:self.img_width
        ]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 5))
        img = cv2.morphologyEx(initial_img, cv2.MORPH_OPEN, kernel)

        contours = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
        contours = [cv2.boundingRect(cnt) for cnt in contours]

        contours = [cnt for cnt in contours if cnt[3] > 20 and cnt[3] < 50]
        contours.sort(key=lambda cnt: cnt[2], reverse=True)  # by width
        l_contours, s_contours = split_contours_by_width(contours)
        l_contours.sort(key=lambda cnt: cnt[1]); s_contours.sort(key=lambda cnt: cnt[1])  # by OY
        check_sum_line = s_contours[-2:]
        check_sum_line.sort(key=lambda cnt: cnt[0])  # by OX

        game_id_line, date_line, *spaced_dashed_numbers = l_contours[-4:]
        game_id_line = (
            game_id_line[0],
            game_id_line[1],
            game_id_line[2] - 327,
            game_id_line[3]
        )
        date_line = (
            date_line[0],
            date_line[1],
            date_line[2] - 167,
            date_line[3]
        )
        self.main_data_contours["game_ids_count"] = len(l_contours[:-3])
        self.main_data_contours["game_id"] = game_id_line
        self.main_data_contours["date"] = date_line
        self.main_data_contours["sum"] = check_sum_line[0]
        self.main_data_contours["spaced_number"] = spaced_dashed_numbers[0]
        self.main_data_contours["dashed_number"] = spaced_dashed_numbers[1]

    def get_game_subtype(self):
        img = cv2.imread("no_needed_repeat_game.png")
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
            if self.check_info["game_type"] == "lotto":
                self.bottom_oy_border_for_table = self.longest_lines["middle_line"]["y"] - 55
            else:
                self.bottom_oy_border_for_table = self.longest_lines["bottom_line"]["y"] - 55

        self._save_debug_img(crop_img, f"subtype/subtype({self.filename}).jpg")
        parsed_subtype = self.get_value_from_image(crop_img, "game_subtype")
        self.check_info["game_subtype"] = parsed_subtype
        return parsed_subtype

    def get_spent_money(self):
        bottom_line = self.longest_lines["bottom_line"]
        sum_contour = self.main_data_contours["sum"]  # (x, y, w, h)
        crop_img = self.wb_blured_img[
            bottom_line["y"] + sum_contour[1]:bottom_line["y"] + sum_contour[1] + sum_contour[3],
            sum_contour[0]:sum_contour[0] + sum_contour[2]
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
        bottom_line = self.longest_lines["bottom_line"]
        dashed_number_contour = self.main_data_contours["dashed_number"]  # (x, y, w, h)
        crop_img = self.wb_blured_img[
            bottom_line["y"] + dashed_number_contour[1]:bottom_line["y"] + dashed_number_contour[1] + dashed_number_contour[3],
            dashed_number_contour[0]:dashed_number_contour[0] + dashed_number_contour[2]
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

    def get_check_spaced_number(self):  # TODO: fix for "#" symbol
        bottom_line = self.longest_lines["bottom_line"]
        spaced_number_contour = self.main_data_contours["spaced_number"]  # (x, y, w, h)
        crop_img = self.wb_blured_img[
            bottom_line["y"] + spaced_number_contour[1]:bottom_line["y"] + spaced_number_contour[1] + spaced_number_contour[3],
            spaced_number_contour[0]:spaced_number_contour[0] + spaced_number_contour[2]
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

    def get_cards(self):
        top_line = self.longest_lines["top_line"]
        bottom_line = self.longest_lines["bottom_line"]

        crop_img = self.wb_blured_img[
            top_line["y"]:bottom_line["y"],
            self.qr_code_info["top_left"][0] - 200:self.qr_code_info["top_right"][0] + 200
        ]
        self._save_debug_img(crop_img, f"cards/cards({self.filename}).jpg")
        self.get_value_from_image(crop_img, "cards")

    def get_table_123_777(self):
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
        min_width = 30 if game_type == "123" else 17
        max_width = 50 if game_type == "123" else 35
        min_height = 35 if game_type == "123" else 35

        sorted_contours = [contour for contour in info_of_contours if contour[2] >= min_width and contour[2] <= max_width and contour[3] >= min_height]

        if game_type == "123":
            if len(sorted_contours) % 3 != 0:
                raise Exception("Not all numbers were found(table_123)")

            # sort contours in right way n11, n21, n31 n12, n22, n32
            sorted_by_OY = sorted(sorted_contours, key=lambda contour: contour[1])

            # group numbers by 3 and sort by OX
            groups_of_numbers = []
            i = 0
            while len(groups_of_numbers) != len(sorted_by_OY) / 3:
                next_three_numbers = sorted_by_OY[i:i + 3]
                next_three_numbers.sort(key=lambda contour: contour[0])  # sort by OX
                groups_of_numbers.append(next_three_numbers)
                i += 3

            for index, group in enumerate(groups_of_numbers):
                self.check_info["table"][f"line_{index + 1}"] = {
                    "regular": []
                }
                for number in group:
                    crop_number = crop_img[
                        number[1]:number[1] + number[3],
                        number[0]:number[0] + number[2]
                    ]

                    table_number = self.get_value_from_image(crop_number, "table")
                    self.check_info["table"][f"line_{index + 1}"]["regular"].append(int(table_number))

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
                stringed_numbers = ""
                for number in line:
                    crop_number = crop_img[
                        number[1]:number[1] + number[3],
                        number[0]:number[0] + number[2]
                    ]
                    table_number = self.get_value_from_image(crop_number, "table")
                    stringed_numbers += table_number

                # merge numbers by two
                i = 0
                while i < len(stringed_numbers):
                    try:
                        make_resulted_number = f"{stringed_numbers[i]}{stringed_numbers[i + 1]}"
                    except:
                        make_resulted_number = f"{stringed_numbers[i]}"
                        print(Style.BRIGHT + Back.RED + Fore.WHITE + f"Not all regular numbers were found in table line")

                    if not self.check_info["table"].get(f"line_{index + 1}", False):
                        self.check_info["table"][f"line_{index + 1}"] = {
                            "regular": []
                        }
                    self.check_info["table"][f"line_{index + 1}"]["regular"].append(int(make_resulted_number))
                    i += 2

        return self.check_info["table"]

    def get_table_lotto(self):
        self.check_info["game_subtype"] = "lotto_regular"
        if self.bottom_oy_border_for_table == 0:
            bottom_oy_border = self.longest_lines["middle_line"]["y"]
        else:
            # finding from top_line to top of subtype if exists
            bottom_oy_border = self.bottom_oy_border_for_table

        crop_img = self.wb_blured_img[
            self.longest_lines["top_line"]["y"]+90:bottom_oy_border,  # 90px it is crop without table header
            0:self.img_width
        ]
        self._save_debug_img(crop_img, f"table/table({self.filename}).jpg")

        inverted_table = cv2.threshold(crop_img, 127, 255, cv2.THRESH_BINARY_INV)[1]
        if self.check_info["game_subtype"] == "lotto_regular":
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 1))
            dilated_table = cv2.dilate(inverted_table, kernel)
            lines_contours = cv2.findContours(dilated_table, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            lines_contours = [cv2.boundingRect(cnt) for cnt in lines_contours]
            sorted_lines_contours = [contour for contour in lines_contours if
                                     contour[2] >= 700 and contour[2] <= 800 and contour[3] >= 40]
            sorted_lines_contours.sort(key=lambda line: line[1])

            for index, line in enumerate(sorted_lines_contours):
                self.check_info["table"][f"line_{index + 1}"] = {
                    "regular": [],
                    "strong": [],
                }
                x, y, w, h = line
                line_img = crop_img[
                    y:y + h,
                    x:x + w
                ]

                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                dilated_line = cv2.dilate(cv2.threshold(line_img, 127, 255, cv2.THRESH_BINARY_INV)[1], kernel)
                number_contours = cv2.findContours(dilated_line, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
                number_contours = [cv2.boundingRect(cnt) for cnt in number_contours]

                sorted_number_contours = [contour for contour in number_contours if
                                          contour[2] >= 10 and contour[3] >= 25]
                sorted_number_contours.sort(key=lambda cnt: cnt[0])
                logging.info(f"Lotto table sorted number contours: {sorted_number_contours}")

                regular_numbers, strong_numbers = self.split_lotto_number_contours(sorted_number_contours)

                # Find regular numbers
                stringed_numbers = ""
                for number in regular_numbers:
                    x, y, w, h = number
                    number_img = line_img[
                        y:y + h,
                        x:x + w
                    ]
                    table_number = self.get_value_from_image(number_img, "table")
                    stringed_numbers += table_number

                logging.info(f"Stringed regular numbers: {stringed_numbers}")

                # merge numbers by two
                i = 0
                while i < len(stringed_numbers):
                    try:
                        make_resulted_number = f"{stringed_numbers[i]}{stringed_numbers[i + 1]}"
                    except:
                        make_resulted_number = f"{stringed_numbers[i]}"
                        print(Style.BRIGHT + Back.RED + Fore.WHITE + f"Not all regular numbers were found in table line")

                    self.check_info["table"][f"line_{index + 1}"]["regular"].append(int(make_resulted_number))
                    i += 2

                # Find strong number
                for number in strong_numbers:
                    x, y, w, h = number
                    number_img = line_img[
                        y:y + h,
                        x:x + w
                    ]
                    table_number = self.get_value_from_image(number_img, "table")
                    if table_number == "(":
                        break
                    self.check_info["table"][f"line_{index + 1}"]["strong"].append(int(table_number))

        # TODO: do parsing for those types
        elif self.check_info["game_subtype"] in ["lotto_strong", "lotto_systematic"]:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            dilated_table = cv2.dilate(inverted_table, kernel)
            number_contours = cv2.findContours(dilated_table, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            number_contours = [cv2.boundingRect(cnt) for cnt in number_contours]

            sorted_number_contours = [contour for contour in number_contours if
                                      contour[2] >= 10 and contour[3] >= 25]
            sorted_number_contours.sort(key=lambda cnt: cnt[0])

            for number in sorted_number_contours:
                x, y, w, h = number
                number_img = crop_img[
                    y:y + h,
                    x:x + w
                ]
                table_number = self.get_value_from_image(number_img, "table")
                self.check_info["table"].append(table_number)

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
        bottom_line = self.longest_lines["bottom_line"]
        date_contour = self.main_data_contours["date"]  # (x, y, w, h)
        crop_img = self.wb_blured_img[
            bottom_line["y"] + date_contour[1]:bottom_line["y"] + date_contour[1] + date_contour[3],
            date_contour[0]:date_contour[0] + date_contour[2]
        ]

        self._save_debug_img(crop_img, f"date/date({self.filename}).jpg")
        numbers = self.get_value_from_image(crop_img, "date")
        if len(numbers) == 12:
            date = f"{numbers[:2]}:{numbers[2:4]}:{numbers[4:6]} {numbers[6:8]}.{numbers[8:10]}.{numbers[10:]}"
            self.check_info["date"] = datetime.datetime.strptime(date, "%H:%M:%S %d.%m.%y")
            return date
        else:
            raise Exception(f"The length of the number is not correct(date)\nNumber: {numbers}")

    def get_game_id(self):
        bottom_line = self.longest_lines["bottom_line"]
        game_id_contour = self.main_data_contours["game_id"]  # (x, y, w, h)
        crop_img = self.wb_blured_img[
            bottom_line["y"] + game_id_contour[1]:bottom_line["y"] + game_id_contour[1] + game_id_contour[3],
            game_id_contour[0]:game_id_contour[0] + game_id_contour[2]
        ]

        self._save_debug_img(crop_img, f"game_id/game_id({self.filename}).jpg")
        numbers = self.get_value_from_image(crop_img, "game_id")
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

            # for bigger border need change height of table_123_numbers.png
            # cropped_img = cv2.copyMakeBorder(
            #     cropped_img,
            #     top=1,
            #     bottom=1,
            #     left=2,
            #     right=2,
            #     borderType=cv2.BORDER_CONSTANT,
            #     value=[255, 255, 255]
            # )

            res = cv2.matchTemplate(img, cropped_img, cv2.TM_CCOEFF_NORMED)
            y, x = np.unravel_index(res.argmax(), res.shape)
            # print(x, y)
            for key, value in d_table_numbers.items():
                if x in range(*value[0]) and y in range(*value[1]):
                    return key

        if data_type == "extra_numbers":
            img = cv2.imread("extra_numbers.png")
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

            res = cv2.matchTemplate(img, cropped_img, cv2.TM_CCOEFF_NORMED)
            y, x = np.unravel_index(res.argmax(), res.shape)
            for key, value in d_extra_numbers.items():
                if x in range(*value[0]) and y in range(*value[1]):
                    return key

        edges = cv2.Canny(cropped_img, 10, 200)
        contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Contours to list of tuples
        info_of_contours = [cv2.boundingRect(contour) for contour in contours]  # (x, y, w, h)

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

        # TODO: in spaced number "01HVF#7R 409361001 000604494" only first part can be with alpha symbols ???
        elif data_type in ["numbers", "date", "sum", "game_id"]:
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
            dilation = cv2.dilate(255 - cropped_img, rect_kernel, iterations=1)

            contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            im2 = cropped_img.copy()
            contours = list(map(lambda cnt: cv2.boundingRect(cnt), contours))
            contours.sort(key=lambda contour: contour[0])

            for index, cnt in enumerate(contours):
                block_x, block_y, block_w, block_h = cnt
                min_block_width = 70 if data_type == "sum" else 100
                if block_w < min_block_width:
                    continue

                block = cropped_img[
                    block_y:block_y + block_h,
                    block_x:block_x + block_w
                ]

                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
                img_dilated = cv2.dilate(cv2.threshold(block, 127, 255, cv2.THRESH_BINARY_INV)[1], kernel)

                if not parse_just_in_numbers and data_type == "numbers":
                    self._save_debug_img(
                        img_dilated,
                        f"horizontal_dilation_numbers/spaced_number(block coord{block_x})({self.filename}).png"
                    )
                else:
                    self._save_debug_img(
                        img_dilated,
                        f"horizontal_dilation_numbers/{data_type}(block coord{block_x})({self.filename}).png"
                    )

                lines_contours = cv2.findContours(img_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
                lines_contours = [cv2.boundingRect(cnt) for cnt in lines_contours]
                lines_contours = list(filter(lambda cnt: cnt[2] > 10, lines_contours))

                if not parse_just_in_numbers and data_type == "numbers" and index == 0:
                    # In first block of spaced number after dilation some can symbols can be in one contour
                    if len(lines_contours) < 8:
                        for cnt in lines_contours:
                            if len(lines_contours) >= 8:
                                break
                            x, y, w, h = cnt
                            if w > 20:
                                count_needed_symbols = 8 - len(lines_contours) + 1  # +1 because we have this contour
                                average_symbol_width = w // count_needed_symbols
                                OX_first_merged_symbol = x
                                for i in range(count_needed_symbols):
                                    lines_contours.append((OX_first_merged_symbol, y, average_symbol_width, h))
                                    OX_first_merged_symbol += average_symbol_width
                        lines_contours = list(filter(lambda cnt: cnt[2] < 23, lines_contours))

                lines_contours.sort(key=lambda cnt: cnt[0])

                for cnt in lines_contours:
                    symbol_x, symbol_y, symbol_w, symbol_h = cnt
                    cropped_contour = block[
                        symbol_y:symbol_y + symbol_h,
                        symbol_x:symbol_x + symbol_w
                    ]

                    symbol_width, symbol_height = cropped_contour.shape[:2]
                    count_non_zero = cv2.countNonZero(cropped_contour)

                    # Find rects in which count of black pixels less than...
                    minimal_count_black_pixels = 40 if data_type == "numbers" else 60
                    if symbol_width * symbol_height - count_non_zero < minimal_count_black_pixels and parse_just_in_numbers:
                        continue

                    cv2.rectangle(im2, (block_x + symbol_x, block_y),
                                  (block_x + symbol_x + symbol_w, block_y + block_h), (0, 0, 0), 1)
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
                    found_symbol_y, found_symbol_x = np.unravel_index(res.argmax(), res.shape)
                    # print(found_symbol_x, found_symbol_y)
                    for key, value in d_all_symbols.items():
                        if found_symbol_x in range(*value[0]) and found_symbol_y in range(*value[1]):
                            result += key

            # or len(result) != 19 and parse_just_in_numbers and data_type == "numbers":
            if len(result) != 26 and not parse_just_in_numbers:
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
            if data_type == "cards":  # TODO: change cards parsing
                img = cv2.imread("cards.png")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                res = cv2.matchTemplate(img, cropped_contour, cv2.TM_CCOEFF_NORMED)
                y, x = np.unravel_index(res.argmax(), res.shape)
                for key, value in cards.items():
                    if x in range(*value[0]) and y in range(*value[1]):
                        card, card_type = key.split("-")
                        self.check_info["cards"][card_type].append(card)

    def get_result(self):
        # All should be called in this way
        try:
            response = self._is_valid_img()
            if response["success"]:
                self.get_coords_of_main_lines()

                logging.info(f"Main check lines: {self.longest_lines}")
                logging.info(f"QR code is found: {self.qr_code_found}")

                game_type = self.get_game_type()

                logging.info(f"Game type: {game_type}")
                print("Game type:", game_type)
                print("Game subtype:", self.get_game_subtype())

                self.get_main_data_contours()

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
        except Exception as ex:
            print(Style.BRIGHT + Back.RED + Fore.WHITE + f"ERROR, CAN`T READ THE DATA FROM IMAGE, {ex}")
            logging.info(f"WERE RAISED UNEXPECTED EXCEPTION: {ex}")


# TODO: remove noise pixels
# TODO: create raise exception in every method after checks
# TODO: find "autofilling" data near subtype

if __name__ == "__main__":
    # Code for testing
    checks_count = 0
    check_folder = "123"  # checks directory
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
