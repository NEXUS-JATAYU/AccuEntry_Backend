from core.database import Base
from sqlalchemy import Column, Integer, String, TIMESTAMP, Boolean , text , Date

class CustomerDetails(Base):
    __tablename__ = "customer_details"

    id = Column(Integer, primary_key=True, index=True)
    c_name = Column(String, nullable=False)
    c_account_type = Column(String, nullable=False)
    c_phone_number = Column(String, unique=True, index=True, nullable=False)
    c_email = Column(String, index=True, nullable=False)
    c_address = Column(String, nullable=False)
    c_occupation = Column(String, nullable=False)
    c_pan = Column(String, unique=True, index=True, nullable=False)
    # c_aadhar = Column(String, unique=True, index=True, nullable=False)
    c_dob = Column(Date, nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), server_default=text('now()'))
    updated_at = Column(TIMESTAMP(timezone=True), onupdate=text('now()'))
    is_active = Column(Boolean, default=True)
