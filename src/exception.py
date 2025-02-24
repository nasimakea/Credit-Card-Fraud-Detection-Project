
import sys
import os

# Automatically add the project root to sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.logger import logging 

def error_message_detail(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename

    error_message = "Error occurred in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )

    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)
        logging.error(self.error_message)  # Log the error when exception is raised

    def __str__(self):
        return self.error_message  

if __name__ == "__main__":
    logging.info("Logging has started")

    try:
        a = 1 / 0  # Intentional division by zero
    except Exception as e:
        # Log the error instead of stopping the program
        error = CustomException(e, sys)
        logging.error("An error occurred: %s", error)

    logging.info("Program completed without crashing.")
    print("Exception module ran successfully!")
