#pragma once
#include "RL4FSStruct.h"
namespace RL4FS {
    class Date
    {
    public:
        Date();
        Date(string date);
        int getDiffSecond(const Date& date);
        bool check();
        bool operator<(const Date& date)const;
        bool operator>(const Date& date)const;
        bool operator==(const Date& date)const;
        bool operator!=(const Date& date)const;
        void makeDate(int year, int month, int day, int hour = 0, int minute = 0, int second = 0);
        friend ostream& operator<< (ostream& out, const Date& date);
        friend istream& operator>> (istream& in, Date& date);
        int getYear();
        int getMonth();
        int getDay();
        int getDayOfWeek();
        int getHour();
        int getMinute();
        int getSecond();
        int isWeekend();

    private:
        int m_year;
        int m_month;
        int m_day;
        int m_hour;
        int m_minute;
        int m_second;
    };
}//RL4FS

