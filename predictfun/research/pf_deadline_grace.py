"""Equal-clock updates precede grace expiry; elapsed grace never exceeds50ms."""
from .pf_inventory_grace import InventoryGrace


class UpdateFirstGrace(InventoryGrace):
    def __init__(self):
        super().__init__(.01,'all',50.)

    def expire(self,now,p):
        for lane,h in list(self.holds.items()):
            if h['deadline']<now:self.close(lane,h['deadline'],'deadline',p)

    def finish_timestamp(self,now,p):
        # Call after ALL events with this exact timestamp. An eligible reopen
        # will already have cleared its grace; an order still off must cancel.
        for lane,h in list(self.holds.items()):
            if h['deadline']<=now:self.close(lane,h['deadline'],'deadline',p)
