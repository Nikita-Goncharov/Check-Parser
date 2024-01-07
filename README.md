## Check parser
***


### Setup
```shell
pip install -r requirements.txt
```

### Usage
```python
from parser import CheckParser
path = "path/to/check_image"
parser = CheckParser(path)
parser.get_result()  # Will print in console all data

# Then you can do "parser.check_info", where will this structure:
# {
#     'qr_code_link': 'https://www.pais.co.il/qr.aspx?q=eNpFibkBACAIxFYKgs8tx+yiFqa4t3UOdpXpPr4dGsKzfhHJpGNJsN5YoZq0Af8EDzAAAAAAAAAA\nAAAAAAAAAAAAAAAAAA==\n', 
#     'date': datetime.datetime(2023, 8, 24, 10, 38, 55), 
#     'game_type': '123', 
#     'game_subtype': '123_regular', 
#     'game_id': '0733623', 
#     'spent_on_ticket': 50.0, 
#     'dashed_number': '2002-048198660-251023', 
#     'spaced_number': '0162HTWP 409351001 000604494', 
#     'extra': False, 
#     'extra_number': '', 
#     'cards': {
#         'hearts': [], 
#         'diamonds': [], 
#         'spades': [], 
#         'clubs': []
#     }, 
#     'table': {
#         'line_1': {'regular': [2, 9, 3]}, 
#         'line_2': {'regular': [1, 9, 4]}, 
#         'line_3': {'regular': [7, 5, 1]}, 
#         'line_4': {'regular': [4, 8, 3]}, 
#         'line_5': {'regular': [1, 4, 8]}
#     }
# }
```

* parser.py - CheckParser parser class with many methods
* elements_coords_in_img.py - file with dicts with coords of symbols from templates images 

### Parsing algorithm:
1) The __init__() method convert image to white and black, save it and create other needed image manipulation
2) Validation image: 
   * Try get QR code, if not - image is not valid - skip it
   * Rotate img if need(if QR code not at the top)
3) Find main lines 777, 123, chance - 2 main lines. Lotto 3 or 4 main lines(Main lines - content lines)
4) Get game type and subtype
5) Find main data contours(main data - game_id, sum, date, spaced and dashed numbers)
6) Get main data
7) Find table(123, 777, lotto(and extra)) or cards(chance)

