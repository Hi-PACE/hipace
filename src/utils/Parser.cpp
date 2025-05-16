/* Copyright 2025
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn
 * License: BSD-3-Clause-LBNL
 */
#include "Parser.H"

namespace Parser {

    void
    DebugPrint () {
        amrex::ParmParse pp("parser", "parser");

        std::vector<std::string> inputs{};

        if (!queryWithParser(pp, "debug_print", inputs)) {
            return;
        }

        amrex::Vector<std::string> local_variables_names{};
        std::vector<std::tuple<double, double, int>> local_variables_bounds{};

        for (auto i=1ul; i!=inputs.size(); ++i) {
            const std::string& s = inputs[i];
            const std::string abort_str = "DebugPrint(): invalid variable format! Must be "
                "\"<var name>=<value>\" or \"<var name>=[<range begin>,<range end>,<num values>]\"."
                " Got: \"" + s + "\"";

            auto pos = s.find_first_of("=");

            if (pos == std::string::npos) {
                amrex::Abort(abort_str);
            }

            std::string varname = s.substr(0, pos);
            std::string expr = s.substr(pos + 1u);
            std::stringstream expr_ss{expr};

            if (expr.size() < 1ul) {
                amrex::Abort(abort_str);
            }

            if (expr_ss.peek() == '[') {
                double range_begin = 0., range_end = 0., num_points = 0.;
                char bracket_open = '[', first_comma = ',', second_comma = ',', bracked_close = ']';

                expr_ss >> bracket_open >> range_begin
                        >> first_comma >> range_end
                        >> second_comma >> num_points
                        >> bracked_close;

                if (bracket_open != '[' || first_comma != ',' ||
                    second_comma != ',' || bracked_close != ']') {
                    amrex::Abort(abort_str);
                }
                local_variables_names.emplace_back(varname);
                local_variables_bounds.emplace_back(range_begin, range_end,
                    static_cast<int>(std::round(num_points)));
            } else {
                double val = 0.;
                expr_ss >> val;
                pp.add(varname.c_str(), val);
            }

            if (expr_ss.fail() ||
                (static_cast<long>(expr_ss.tellg()) != static_cast<long>(expr.size()) &&
                static_cast<long>(expr_ss.tellg()) != -1l)) {
                amrex::Abort(abort_str);
            }
        }

        amrex::Parser func_parser{};

        if (local_variables_names.size() == 0) {
            auto exe = makeFunctionWithParser<0>(inputs[0], func_parser, {}, pp);
            std::cout << "\nParser Debug Print \"" << inputs[0] << "\" = "
                      << exe() << "\n" << std::endl;
        } else if (local_variables_names.size() == 1) {
            auto exe = makeFunctionWithParser<1>(inputs[0], func_parser, local_variables_names, pp);
            std::cout << "\nParser Debug Print \"" << inputs[0] << "\" = [";
            auto [lo, hi, n] = local_variables_bounds[0];
            const double dx = (hi-lo)/(n-1);
            for (int i=0; i<n; ++i) {
                double x = lo + i * dx;
                const double val = exe(x);
                if (i > 0) {
                    std::cout << ", ";
                }
                std::cout << val;
            }
            std::cout << "]\n" << std::endl;
        } else if (local_variables_names.size() == 2) {
            auto exe = makeFunctionWithParser<2>(inputs[0], func_parser, local_variables_names, pp);
            std::cout << "\nParser Debug Print \"" << inputs[0] << "\" = [";
            auto [lox, hix, nx] = local_variables_bounds[0];
            auto [loy, hiy, ny] = local_variables_bounds[1];
            const double dx = (hix-lox)/(nx-1);
            const double dy = (hiy-loy)/(ny-1);
            for (int j=0; j<ny; ++j) {
                const double y = loy + j * dy;
                if (j > 0) {
                    std::cout << ",\n";
                }
                std::cout << "[";
                for (int i=0; i<nx; ++i) {
                    const double x = lox + i * dx;
                    const double val = exe(x, y);
                    if (i > 0) {
                        std::cout << ", ";
                    }
                    std::cout << val;
                }
                std::cout << "]";
            }
            std::cout << "]\n" << std::endl;
        } else if (local_variables_names.size() == 3) {
            auto exe = makeFunctionWithParser<3>(inputs[0], func_parser, local_variables_names, pp);
            std::cout << "\nParser Debug Print \"" << inputs[0] << "\" = [";
            auto [lox, hix, nx] = local_variables_bounds[0];
            auto [loy, hiy, ny] = local_variables_bounds[1];
            auto [loz, hiz, nz] = local_variables_bounds[2];
            const double dx = (hix-lox)/(nx-1);
            const double dy = (hiy-loy)/(ny-1);
            const double dz = (hiz-loz)/(nz-1);
            for (int k=0; k<nz; ++k) {
                const double z = loz + k * dz;
                if (k > 0) {
                    std::cout << ",\n\n";
                }
                std::cout << "[";
                for (int j=0; j<ny; ++j) {
                    const double y = loy + j * dy;
                    if (j > 0) {
                        std::cout << ",\n";
                    }
                    std::cout << "[";
                    for (int i=0; i<nx; ++i) {
                        const double x = lox + i * dx;
                        const double val = exe(x, y, z);
                        if (i > 0) {
                            std::cout << ", ";
                        }
                        std::cout << val;
                    }
                    std::cout << "]";
                }
                std::cout << "]";
            }
            std::cout << "]\n" << std::endl;
        } else if (local_variables_names.size() == 4) {
            auto exe = makeFunctionWithParser<4>(inputs[0], func_parser, local_variables_names, pp);
            std::cout << "\nParser Debug Print \"" << inputs[0] << "\" = [";
            auto [lox, hix, nx] = local_variables_bounds[0];
            auto [loy, hiy, ny] = local_variables_bounds[1];
            auto [loz, hiz, nz] = local_variables_bounds[2];
            auto [low, hiw, nw] = local_variables_bounds[3];
            const double dx = (hix-lox)/(nx-1);
            const double dy = (hiy-loy)/(ny-1);
            const double dz = (hiz-loz)/(nz-1);
            const double dw = (hiw-low)/(nw-1);
            for (int l=0; l<nw; ++l) {
                const double w = low + l * dw;
                if (l > 0) {
                    std::cout << ",\n\n\n";
                }
                std::cout << "[";
                for (int k=0; k<nz; ++k) {
                    const double z = loz + k * dz;
                    if (k > 0) {
                        std::cout << ",\n\n";
                    }
                    std::cout << "[";
                    for (int j=0; j<ny; ++j) {
                        const double y = loy + j * dy;
                        if (j > 0) {
                            std::cout << ",\n";
                        }
                        std::cout << "[";
                        for (int i=0; i<nx; ++i) {
                            const double x = lox + i * dx;
                            const double val = exe(x, y, z, w);
                            if (i > 0) {
                                std::cout << ", ";
                            }
                            std::cout << val;
                        }
                        std::cout << "]";
                    }
                    std::cout << "]";
                }
                std::cout << "]";
            }
            std::cout << "]\n" << std::endl;
        } else {
            amrex::Abort("DebugPrint(): Only supports up to 4 variables");
        }
    }

}
