#include "RL4FS/Date.h"
namespace RL4FS {
    
    Date::Date() :m_year(0), m_month(0), m_day(0), m_hour(0), m_minute(0), m_second(0) {};
    Date::Date(string date) :m_year(0), m_month(0), m_day(0), m_hour(0), m_minute(0), m_second(0)
    {
        //1970-01-06 10:45:02
        string sdate;
        int indexdate = 0;
        for (auto& onechar : date)
        {
            if (onechar == '-' || onechar == ' ')
            {
                switch (indexdate)
                {
                case 0: m_year = stoi(sdate); break;
                case 1: m_month = stoi(sdate); break;
                case 2: m_day = stoi(sdate); break;
                }
                sdate.clear();
                indexdate++;
            }
            else if (onechar == ':')
            {
                switch (indexdate)
                {
                case 3: m_hour = stoi(sdate); break;
                case 4: m_minute = stoi(sdate); break;
                case 5: m_second = stoi(sdate); break;
                }
                sdate.clear();
                indexdate++;
            }
            else sdate += onechar;
        }
        if (m_day > 0)m_second = stoi(sdate);
        else m_day = stoi(sdate);
        if (!check())
        {
            m_year = 0;
            m_month = 0;
            m_day = 0;
            m_hour = 0;
            m_minute = 0;
            m_second = 0;
            LOG(WARNING) << date << " is not correct date!";
        }
    }
    int Date::getDiffSecond(const Date& date)
    {
        Date date1, date2;
        if (*this == date) return 0;
        else if (*this > date)
            date1 = *this, date2 = date;
        else date1 = date, date2 = *this;
        int diff = 0;
        int diffmin = 60;
        int diffhour = diffmin * 60;
        int diffday = diffhour * 24;
        int diffmonth = diffday * 30;
        int diffyear = diffmonth * 12;
        diff += date1.getSecond() - date2.getSecond();
        diff += (date1.getMinute() - date2.getMinute()) * diffmin;
        diff += (date1.getHour() - date2.getHour()) * diffhour;
        diff += (date1.getDay() - date2.getDay()) * diffday;
        diff += (date1.getMonth() - date2.getMonth()) * diffmonth;
        diff += (date1.getYear() - date2.getYear()) * diffyear;
        return diff;
    }
    bool Date::check()
    {
        if (m_year < 0)return false;
        if (m_month > 12 || m_month <= 0)return false;
        if (m_day > 31 || m_day <= 0)return false;
        if (m_hour >= 24 || m_hour < 0)return false;
        if (m_minute >= 60 || m_minute < 0)return false;
        if (m_second >= 60 || m_second < 0)return false;
        return true;
    }
    bool Date::operator<(const Date& date)const
    {
        if (*this == date)return false;
        if (m_year == date.m_year)
        {
            if (m_month == date.m_month)
            {
                if (m_day == date.m_day)
                {
                    if (m_hour == date.m_hour)
                    {
                        if (m_minute == date.m_minute)
                        {
                            return m_second < date.m_second;
                        }
                        return m_minute < date.m_minute;
                    }
                    return m_hour < date.m_hour;
                }
                return m_day < date.m_day;
            }
            return m_month < date.m_month;
        }
        return m_year < date.m_year;
    }
    bool Date::operator>(const Date& date)const
    {
        if (*this == date)return false;
        return !(*this < date);
    }
    bool Date::operator==(const Date& date)const
    {
        if (m_year == date.m_year && m_month == date.m_month && m_day == date.m_day && m_hour == date.m_hour && m_minute == date.m_minute && m_second == date.m_second)
            return true;
        return false;
    }
    bool Date::operator!=(const Date& date)const
    {
        return !(*this == date);
    }
    void Date::makeDate(int year, int month, int day, int hour, int minute, int second)
    {
        m_year = year;
        m_month = month;
        m_day = day;
        m_hour = hour;
        m_minute = minute;
        m_second = second;
    }
    ostream& operator<< (ostream& out, const Date& date)
    {
        out << date.m_year << "-" << date.m_month << "-" << date.m_day <<
            " " << date.m_hour << ":" << date.m_minute << ":" << date.m_second;
        return out;
    }
    istream& operator>> (istream& in, Date& date)
    {
        string val1, val2;
        in >> val1 >> val2;
        val1 += "";
        val1 += val2;
        date = Date(val1);
        return in;
    }
    int Date::getYear()
    {
        return m_year;
    }
    int Date::getMonth()
    {
        return m_month;
    }
    int Date::getDay()
    {
        return m_day;
    }
    int Date::getDayOfWeek()
    {
        if (m_month == 1 || m_month == 2) 
        {
            m_month += 12;
            m_year--;
        }
        int Week = (m_day + 2 * m_month + 3 * (m_month + 1) / 5 + m_year + m_year / 4 - m_year / 100 + m_year / 400) % 7;
        return Week;
    }
    int Date::getHour()
    {
        return m_hour;
    }
    int Date::getMinute()
    {
        return m_minute;
    }
    int Date::getSecond()
    {
        return m_second;
    }
    int Date::isWeekend()
    {
        if (getDayOfWeek() == 0 || getDayOfWeek() == 6)
            return 1;
        else
            return 0;
    }

}//RL4FS

