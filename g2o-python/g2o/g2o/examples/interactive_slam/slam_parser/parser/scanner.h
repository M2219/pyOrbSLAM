#ifndef SCANNER_H
#define SCANNER_H

#ifndef YY_DECL

#define YY_DECL                                              \
  slam_parser::Parser::token_type slam_parser::Scanner::lex( \
      slam_parser::Parser::semantic_type* yylval,            \
      slam_parser::Parser::location_type* yylloc)
#endif

#ifndef __FLEX_LEXER_H
#define yyFlexLexer SlamFlexLexer
#include "FlexLexer.h"
#undef yyFlexLexer
#endif

#include "bison_parser.h"

namespace slam_parser {

class Scanner : public SlamFlexLexer {
 public:
  /** Create a new scanner object. The streams arg_yyin and arg_yyout default
   * to cin and cout, but that assignment is only made when initializing in
   * yylex(). */
  explicit Scanner(std::istream* arg_yyin = nullptr,
                   std::ostream* arg_yyout = nullptr);

  /** Required for virtual functions */
  ~Scanner() override = default;

  /** This is the main lexing function. It is generated by flex according to
   * the macro declaration YY_DECL above. The generated bison parser then
   * calls this virtual function to fetch new tokens. */
  virtual Parser::token_type lex(Parser::semantic_type* yylval,
                                 Parser::location_type* yylloc);

  /** Enable debug output (via arg_yyout) if compiled into the scanner. */
  void set_debug(bool b);
};

}  // namespace slam_parser

#endif