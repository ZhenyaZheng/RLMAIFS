#pragma once
#include <string>
using std::string;
namespace RL4FS {
    class MyExcept: public std::exception
    {
    public:
        MyExcept(string msg)
        {
            m_msg = msg;
        }
        string getMsg()
        {
            return m_msg + string(what());
        }
    private:
        string m_msg;
    };
}//RL4FS