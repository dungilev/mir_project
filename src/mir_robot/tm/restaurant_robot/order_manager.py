from dataclasses import dataclass, field


@dataclass
class Order:
    items: list = field(default_factory=list)

    def get_total(self):
        return sum(item.get("price", 0) for item in self.items)


class OrderManager:
    def __init__(self):
        self.orders = {}

    def _get(self, table):
        if table not in self.orders:
            self.orders[table] = Order()
        return self.orders[table]

    def add_items_to_table(self, table, items):
        order = self._get(table)
        order.items.extend(items)
        return True, [f"Đã thêm {len(items)} món cho bàn {table}."]

    def get_order_summary(self, table):
        order = self.orders.get(table)
        if not order or not order.items:
            return None, ""
        qty = len(order.items)
        return order, f"{qty} nước lọc"

    def cancel_items(self, table):
        if table in self.orders:
            self.orders[table].items = []

    def confirm_order(self, table):
        return True

    def mark_preparing(self, table):
        return True

    def mark_ready(self, table):
        return True

    def mark_delivering(self, table):
        return True

    def mark_delivered(self, table):
        return True

    def pay_and_close(self, table):
        order = self.orders.pop(table, None)
        total = order.get_total() if order else 0
        return True, total

    def cleanup_table(self, table):
        self.orders.pop(table, None)
