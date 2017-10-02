#pragma once

#include <iomanip>
#include <plog/Appenders/ColorConsoleAppender.h>
#include <plog/Log.h>
#include <plog/Record.h>
#include <plog/Util.h>

namespace plog {

template <class T>
Record& operator<<(Record& record,
    const std::vector<T>& vec) // Implement a stream operator for our type.
{
    record << "<";
    for (size_t i = 0; i < vec.size() - 1; i++) {
        record << vec[i] << ";";
    }
    record << vec.back() << ">";
    return record;
}

template <class T> Record& operator<<(Record& record, const std::pair<T, T>& p)
{
    record << "(" << p.first << "," << p.second << ")";
    return record;
}
}

namespace plog {
class MyFormatter {
public:
    static util::nstring header() // This method returns a header for a new
    // file. In our case it is empty.
    {
        return util::nstring();
    }

    static util::nstring format(
        const Record& record) // This method returns a string from a record.
    {
        tm t;
        util::localtime_s(&t, &record.getTime().time);

        util::nstringstream ss;
        ss << t.tm_year + 1900 << "-" << std::setfill(PLOG_NSTR('0'))
           << std::setw(2) << t.tm_mon + 1 << PLOG_NSTR("-")
           << std::setfill(PLOG_NSTR('0')) << std::setw(2) << t.tm_mday
           << PLOG_NSTR(" ");
        ss << std::setfill(PLOG_NSTR('0')) << std::setw(2) << t.tm_hour
           << PLOG_NSTR(":") << std::setfill(PLOG_NSTR('0')) << std::setw(2)
           << t.tm_min << PLOG_NSTR(":") << std::setfill(PLOG_NSTR('0'))
           << std::setw(2) << t.tm_sec << PLOG_NSTR(".")
           << std::setfill(PLOG_NSTR('0')) << std::setw(3)
           << record.getTime().millitm << PLOG_NSTR(" ");
        ss << std::setfill(PLOG_NSTR(' ')) << std::setw(5) << std::left
           << severityToString(record.getSeverity()) << PLOG_NSTR(" ");
        ss << record.getMessage() << PLOG_NSTR("\n");

        return ss.str();
    }
};
}

namespace logging {
static void init()
{
    static plog::ColorConsoleAppender<plog::MyFormatter> consoleAppender;
    plog::init(plog::verbose, &consoleAppender);
}
}