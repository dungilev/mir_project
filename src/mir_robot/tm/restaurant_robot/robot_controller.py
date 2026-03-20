class RobotController:
    def __init__(self, init_ros_node=True):
        self.init_ros_node = init_ros_node

    def navigate_to_table(self, table_number):
        return True, f"Đi tới bàn {table_number}"

    def navigate_to_kitchen(self):
        return True, "Đi tới bếp"

    def navigate_to_home(self):
        return True, "Về vị trí trực"

    def cancel_navigation(self):
        return True

    def is_navigation_done(self):
        return True

    def stop(self):
        return True
