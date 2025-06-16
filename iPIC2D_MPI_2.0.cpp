#include <iostream>
#include <cmath>
#include <random>
#include <vector>
#include <mpi.h>
#include <algorithm>
#include <array>
 
struct Particle {
    double px;
    double py;
    double vx; 
    double vy; 
    int pm; 
};

void Poisson_solver(std::vector<std::vector<double>>& phi, const std::vector<std::vector<double>>& rho, 
    int NGx_domain, int NGy_domain, double dx, double dy, int left, int right, int up, int down, 
    MPI_Comm cart_comm, int max_iteration = 10000, double tol = 1e-6) {

    std::vector<std::vector<double>> phi_new = phi;

    std::vector<double> send_up_buffer_phi(NGx_domain);
    std::vector<double> send_down_buffer_phi(NGx_domain);
    std::vector<double> send_left_buffer_phi(NGy_domain);
    std::vector<double> send_right_buffer_phi(NGy_domain); 
    
    std::vector<double> recv_down_buffer_phi(NGx_domain);
    std::vector<double> recv_up_buffer_phi(NGx_domain);
    std::vector<double> recv_right_buffer_phi(NGy_domain);
    std::vector<double> recv_left_buffer_phi(NGy_domain); 

    for (int iter = 0; iter < max_iteration; ++iter) {
        if (up != MPI_PROC_NULL) {
            for (int i = 1; i <= NGx_domain; ++i) {
                send_up_buffer_phi[i - 1] = phi[i][NGy_domain]; 
            }

            MPI_Sendrecv(send_up_buffer_phi.data(), NGx_domain, MPI_DOUBLE, up, 0, recv_down_buffer_phi.data(), NGx_domain, MPI_DOUBLE, down, 0, cart_comm, MPI_STATUS_IGNORE);
        }

        if (down != MPI_PROC_NULL) {
            for (int i = 1; i <= NGx_domain; ++i) {
                send_down_buffer_phi[i - 1] = phi[i][1];
            }

            MPI_Sendrecv(send_down_buffer_phi.data(), NGx_domain, MPI_DOUBLE, down, 1, recv_up_buffer_phi.data(), NGx_domain, MPI_DOUBLE, up, 1, cart_comm, MPI_STATUS_IGNORE);
        }

        if (up != MPI_PROC_NULL) { 
            for (int i = 1; i <= NGx_domain; ++i) {
                phi[i][NGy_domain + 1] = recv_up_buffer_phi[i - 1];
            }
        }

        if (down != MPI_PROC_NULL) { 
            for (int i = 1; i <= NGx_domain; ++i) {
                phi[i][0] = recv_down_buffer_phi[i-1];
            }
        }

        if (left != MPI_PROC_NULL) { 
            for (int j = 1; j <= NGy_domain; ++j) {
                send_left_buffer_phi[j - 1] = phi[1][j];
            }

            MPI_Sendrecv(send_left_buffer_phi.data(), NGy_domain, MPI_DOUBLE, left, 2, recv_right_buffer_phi.data(), NGy_domain, MPI_DOUBLE, right, 2, cart_comm, MPI_STATUS_IGNORE);
        }

        if (right != MPI_PROC_NULL) { 
            for (int j = 1; j <= NGy_domain; ++j) {
                send_right_buffer_phi[j - 1] = phi[NGx_domain][j];
            }

            MPI_Sendrecv(send_right_buffer_phi.data(), NGy_domain, MPI_DOUBLE, right, 3, recv_left_buffer_phi.data(), NGy_domain, MPI_DOUBLE, left, 3, cart_comm, MPI_STATUS_IGNORE);
        }

        if (left != MPI_PROC_NULL) {
            for (int j = 1; j <= NGy_domain; ++j) {
                phi[0][j] = recv_left_buffer_phi[j - 1];
            }
        }

        if (right != MPI_PROC_NULL) { 
            for (int j = 1; j <= NGy_domain; ++j) {
                phi[NGx_domain + 1][j] = recv_right_buffer_phi[j - 1];
            }
        }

        if (left == MPI_PROC_NULL) {
            for (int j = 1; j <= NGy_domain; ++j) {
                phi[0][j] = phi[1][j]; 
            }
        }

        if (right == MPI_PROC_NULL) {
            for (int j = 1; j <= NGy_domain; ++j) {
                phi[NGx_domain + 1][j] = phi[NGx_domain][j]; 
            }
        }

        if (up == MPI_PROC_NULL) {
            for (int i = 1; i <= NGx_domain; ++i) {
                phi[i][NGy_domain + 1] = phi[i][NGy_domain]; 
            }
        }

        if (down == MPI_PROC_NULL) {
            for (int i = 1; i <= NGx_domain; ++i) {
                phi[i][0] = phi[i][1]; 
            }
        }

        double local_error = 0; 

        for (int i = 1; i <= NGx_domain; ++i) { 
            for (int j = 1; j <= NGy_domain; ++j) {
                phi_new[i][j] = (0.25) * ((phi[i + 1][j]) + (phi[i - 1][j]) + (phi[i][j + 1]) + (phi[i][j - 1]) + ((dx * dx) * rho[i][j])); 
                double value_change = std::abs(phi_new[i][j] - phi[i][j]);
                if (value_change > local_error) {
                    local_error = value_change;
                }
            }
        }

        double global_error; 
        MPI_Allreduce(&local_error, &global_error, 1, MPI_DOUBLE, MPI_MAX, cart_comm);

        if (global_error < tol) {
            break;
        }
        phi = phi_new; 
    }
}

void Field_solver(std::vector<std::vector<double>> &phi, std::vector<std::vector<double>> &E_x, std::vector<std::vector<double>> &E_y,
    int NGx_domain, int NGy_domain, double dx, double dy) {

    for (int i = 1; i <= NGx_domain; ++i) { 
        for (int j = 1; j <= NGy_domain; ++j) {
            E_x[i][j] = -(phi[i +1][j] - phi[i -1][j]) / (2 * dx);
            E_y[i][j] = -(phi[i][j +1] - phi[i][j -1]) / (2 * dy); 
        }
    }
}

struct Sphere {
    double x, y;
    double radius;
    double charge;
};

int main(int argc, char** argv) {

    double Lx = 2 * M_PI; 
    double Ly = 2 * M_PI;

    int NGx = 32;
    int NGy = 32;  

    double dx = Lx / NGx; 
    double dy = Ly / NGy; 

    double dt = 0.5; 
    int NT = 50;

    int N_e = 1000; 
    int N_i = 1000;

    double WP = 1.0;

    const double q_e = -1.0;
    const double m_e = 1.0;
    const double qm_e = q_e / m_e;

    const double q_i = 1.0; 
    const double m_i = 1836.0; 
    const double qm_i = q_i / m_i; 

    const double k = 1.0 / (4.0 * M_PI);

    double Q_e = (WP * WP) / ((qm_e * N_e) / (Lx * Ly));
    double Q_i = -Q_e;

    double VT = 0.0;
    double V0 = 0.2;
    double XP1 = 1.0;

    double V1 = 0.0;
    int mode = 1;

    MPI_Init(&argc, &argv);

    int rank;
    int size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int dims[2] = {0, 0};
    MPI_Dims_create(size, 2, dims);

    int periods[2] = {0, 0};
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cart_comm);

    int coords[2];
    MPI_Cart_coords(cart_comm, rank, 2, coords);

    int left;
    int right;
    int up;
    int down;
    MPI_Cart_shift(cart_comm, 0, 1, &left, &right);
    MPI_Cart_shift(cart_comm, 1, 1, &down, &up);

    std::cout << "The rank to the left of " << rank << " is " << left << std::endl;
    std::cout << "The rank to the right of " << rank << " is " << right << std::endl;


    int NGx_domain = NGx / dims[0];
    int NGy_domain = NGy / dims[1];

    double x0 = coords[0] * NGx_domain * dx;
    double y0 = coords[1] * NGy_domain * dy;
    double xF = (coords[0] + 1) * NGx_domain * dx;
    double yF = (coords[1] + 1) * NGy_domain * dy;

    int Ne_domain = N_e / size;
    int Ni_domain = N_i / size;

    Sphere sphere;
    sphere.x = M_PI;
    sphere.y = M_PI;
    sphere.radius = M_PI / 4;
    sphere.charge = 0.0;

    std::vector <Particle> electrons;
    std::vector <Particle> ions;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> normal_dist(0.0, 1.0);

    std::vector<std::vector<double>> rho(NGx_domain + 2, std::vector<double>(NGy_domain + 2, 0.0));
    std::vector<std::vector<double>> phi(NGx_domain + 2, std::vector<double>(NGy_domain + 2, 0.0));
    std::vector<std::vector<double>> E_x(NGx_domain + 2, std::vector<double>(NGy_domain + 2, 0.0));
    std::vector<std::vector<double>> E_y(NGx_domain + 2, std::vector<double>(NGy_domain + 2, 0.0));

    std::uniform_real_distribution<double> uniform_x(x0, xF);
    std::uniform_real_distribution<double> uniform_y(y0, yF);

    for (int i = 0; i < Ne_domain; ++i) {
        Particle e; 
        e.px = uniform_x(gen);
        e.py = uniform_y(gen);
        e.vx = VT * normal_dist(gen);
        e.vy = VT * normal_dist(gen);
        electrons.push_back(e);   
    }

    for (int i = 0; i < Ni_domain; ++i) {
        Particle ion;
        ion.px = uniform_x(gen);
        ion.py = uniform_y(gen);
        ion.vx = VT * normal_dist(gen);
        ion.vy = VT * normal_dist(gen);
        ions.push_back(ion);
    }

    for (size_t i = 0; i < electrons.size(); ++i) {
        electrons[i].pm = 1 - 2 * (i % 2);
        electrons[i].vx = electrons[i].vx + (V0 * electrons[i].pm);
        electrons[i].vx = electrons[i].vx + (V1 * sin(2 * M_PI * electrons[i].px / (Lx * mode)));
        electrons[i].px = electrons[i].px + (XP1 * (Lx/N_e) * sin(2 * M_PI * electrons[i].px / (Lx * mode)));
    }

    for (size_t i = 0; i < ions.size(); ++i) {
        ions[i].pm = 1 - 2 * (i % 2);
        ions[i].vx = ions[i].vx + (V0 * ions[i].pm);
        ions[i].vx = ions[i].vx + (V1 * sin(2 * M_PI * ions[i].px / (Lx * mode)));
        ions[i].px = ions[i].px + (XP1 * (Lx/N_e) * sin(2 * M_PI * ions[i].px / (Lx * mode)));
    }

    for (int i = 0; i < static_cast<int>(electrons.size()); ++i) {
        double px_local = electrons[i].px - x0;
        double py_local = electrons[i].py - y0;

        double px_grid = px_local / dx;
        double py_grid = py_local / dy;

        int i1 = static_cast<int>(std::floor(px_grid - 0.5));
        int j1 = static_cast<int>(std::floor(py_grid - 0.5));

        i1 = std::max(0, std::min(i1, NGx_domain - 2));
        j1 = std::max(0, std::min(j1, NGy_domain - 2));

        std::array<int, 2> ix = {i1, i1 + 1};
        std::array<int, 2> jy = {j1, j1 + 1};

        double frazi1 = px_grid - i1 - 0.5;
        double frazj1 = py_grid - j1 - 0.5;

        frazi1 = std::min(std::max(frazi1, 0.0), 1.0);
        frazj1 = std::min(std::max(frazj1, 0.0), 1.0);

        double w_ileft = 1 - frazi1;
        double w_iright = frazi1; 
        double w_jdown = 1 - frazj1; 
        double w_jup = frazj1;

        std::array<double, 2> frazx = {w_ileft, w_iright};
        std::array<double, 2> frazy = {w_jdown, w_jup};

        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                int indexi = ix[i] + 1; 
                int indexj = jy[j] + 1; 
                if (indexi <= NGx_domain && indexi >= 1 && indexj <= NGy_domain && indexj >=1) {
                    rho[indexi][indexj] = rho[indexi][indexj] + ((Q_e / (dx * dy)) * frazx[i] * frazy[j]);
                }
            }
        }
    }

    for (int i = 0; i < static_cast<int>(ions.size()); ++i) {
        double px_local = ions[i].px - x0;
        double py_local = ions[i].py - y0;

        double px_grid = px_local / dx;
        double py_grid = py_local / dy;

        int i1 = static_cast<int>(std::floor(px_grid - 0.5));
        int j1 = static_cast<int>(std::floor(py_grid - 0.5));

        i1 = std::max(0, std::min(i1, NGx_domain - 2));
        j1 = std::max(0, std::min(j1, NGy_domain - 2));

        std::array<int, 2> ix = {i1, i1 + 1};
        std::array<int, 2> jy = {j1, j1 + 1};

        double frazi1 = px_grid - i1 - 0.5;
        double frazj1 = py_grid - j1 - 0.5;

        frazi1 = std::min(std::max(frazi1, 0.0), 1.0);
        frazj1 = std::min(std::max(frazj1, 0.0), 1.0);

        double w_ileft = 1 - frazi1;
        double w_iright = frazi1; 
        double w_jdown = 1 - frazj1; 
        double w_jup = frazj1;

        std::array<double, 2> frazx = {w_ileft, w_iright};
        std::array<double, 2> frazy = {w_jdown, w_jup};

        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                int indexi = ix[i] + 1; 
                int indexj = jy[j] + 1; 
                if (indexi <= NGx_domain && indexi >= 1 && indexj <= NGy_domain && indexj >=1) {
                    rho[indexi][indexj] = rho[indexi][indexj] + ((Q_i / (dx * dy)) * frazx[i] * frazy[j]);
                }
            }
        }
    }

    std::vector<double> send_buffer_left_grid_initial(NGy_domain); 
    std::vector<double> send_buffer_right_grid_initial(NGy_domain); 
    std::vector<double> send_buffer_up_grid_initial(NGx_domain); 
    std::vector<double> send_buffer_down_grid_initial(NGx_domain); 

    std::vector<double> recv_buffer_left_grid_initial(NGy_domain); 
    std::vector<double> recv_buffer_right_grid_initial(NGy_domain);
    std::vector<double> recv_buffer_up_grid_initial(NGx_domain); 
    std::vector<double> recv_buffer_down_grid_initial(NGx_domain);

    if (left != MPI_PROC_NULL) {
        for (int j = 1; j <= NGy_domain; ++j) { 
            send_buffer_left_grid_initial[j - 1] = rho[1][j];
        }

        MPI_Sendrecv(send_buffer_left_grid_initial.data(), NGy_domain, MPI_DOUBLE, left, 0, recv_buffer_right_grid_initial.data(), NGy_domain, MPI_DOUBLE, right, 0, cart_comm, MPI_STATUS_IGNORE);
    }

    if (right != MPI_PROC_NULL) {
        for (int j = 1; j <= NGy_domain; ++j) {
            send_buffer_right_grid_initial[j - 1] = rho[NGx_domain][j];
        }

        MPI_Sendrecv(send_buffer_right_grid_initial.data(), NGy_domain, MPI_DOUBLE, right, 1, recv_buffer_left_grid_initial.data(), NGy_domain, MPI_DOUBLE, left, 1, cart_comm, MPI_STATUS_IGNORE);
    }

    if (up != MPI_PROC_NULL) {
        for (int i = 1; i <= NGx_domain; ++i) {
            send_buffer_up_grid_initial[i - 1] = rho[i][NGy_domain];
        }

        MPI_Sendrecv(send_buffer_up_grid_initial.data(), NGx_domain, MPI_DOUBLE, up, 2, recv_buffer_down_grid_initial.data(), NGx_domain, MPI_DOUBLE, down, 2, cart_comm, MPI_STATUS_IGNORE);
    }

    if (down != MPI_PROC_NULL) {
        for (int i = 1; i <= NGx_domain; ++i) {
            send_buffer_down_grid_initial[i - 1] = rho[i][1]; 
        }

        MPI_Sendrecv(send_buffer_down_grid_initial.data(), NGx_domain, MPI_DOUBLE, down, 3, recv_buffer_up_grid_initial.data(), NGx_domain, MPI_DOUBLE, up, 3, cart_comm, MPI_STATUS_IGNORE);
    
    }

    if (left != MPI_PROC_NULL) {
        for (int j = 1; j <= NGy_domain; ++j) {
            rho[0][j] = recv_buffer_left_grid_initial[j - 1];
        }
    }

    if (right != MPI_PROC_NULL) {
        for (int j = 1; j <= NGy_domain; ++j) {
            rho[NGx_domain + 1][j] = recv_buffer_right_grid_initial[j - 1];
        }
    }

    if (up != MPI_PROC_NULL) {
        for (int i = 1; i <= NGx_domain; ++i) {
            rho[i][NGy_domain + 1] = recv_buffer_up_grid_initial[i - 1];
        }
    }

    if (down != MPI_PROC_NULL) {
        for (int i = 1; i <= NGx_domain; ++i) {
            rho[i][0] = recv_buffer_down_grid_initial[i - 1];
        }
    }

    Poisson_solver(phi, rho, NGx_domain, NGy_domain, dx, dy, left, right, up, down, cart_comm);

    Field_solver(phi, E_x, E_y, NGx_domain, NGy_domain, dx, dy);

    for (int i = 0; i < static_cast<int>(electrons.size()); ++i) {
        double px_local = electrons[i].px - x0;
        double py_local = electrons[i].py - y0;

        double px_grid = px_local / dx;
        double py_grid = py_local / dy;

        int i1 = static_cast<int>(std::floor(px_grid - 0.5));
        int j1 = static_cast<int>(std::floor(py_grid - 0.5));

        std::array<int, 2> ix = {i1, i1 + 1};
        std::array<int, 2> jy = {j1, j1 + 1};

        double frazi1 = px_grid - i1 - 0.5;
        double frazj1 = py_grid - j1 - 0.5;

        double w_ileft = 1 - frazi1;
        double w_iright = frazi1; 
        double w_jdown = 1 - frazj1; 
        double w_jup = frazj1; 

        std::array<double, 2> frazx = {w_ileft, w_iright};
        std::array<double, 2> frazy = {w_jdown, w_jup};

        double Ex_electron = 0.0;
        double Ey_electron = 0.0;

        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                int indexi = ix[i] + 1;
                int indexj = jy[j] + 1;

                if (indexi >= 1 && indexi <= NGx_domain && indexj >= 1 && indexj <= NGy_domain) {
                    Ex_electron += E_x[indexi][indexj] * frazx[i] * frazy[j];
                    Ey_electron += E_y[indexi][indexj] * frazx[i] * frazy[j];
                }
            }
        }

        electrons[i].vx += (qm_e * Ex_electron * (dt / 2));
        electrons[i].vy += (qm_e * Ey_electron * (dt / 2));
    }

    for (int i = 0; i < static_cast<int>(ions.size()); ++i) {
        double px_local = ions[i].px - x0;
        double py_local = ions[i].py - y0;

        double px_grid = px_local / dx;
        double py_grid = py_local / dy;

        int i1 = static_cast<int>(std::floor(px_grid - 0.5));
        int j1 = static_cast<int>(std::floor(py_grid - 0.5));

        std::array<int, 2> ix = {i1, i1 + 1};
        std::array<int, 2> jy = {j1, j1 + 1};

        double frazi1 = px_grid - i1 - 0.5;
        double frazj1 = py_grid - j1 - 0.5;

        double w_ileft = 1 - frazi1;
        double w_iright = frazi1; 
        double w_jdown = 1 - frazj1; 
        double w_jup = frazj1; 

        std::array<double, 2> frazx = {w_ileft, w_iright};
        std::array<double, 2> frazy = {w_jdown, w_jup};

        double Ex_ion = 0.0;
        double Ey_ion = 0.0;

        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                int indexi = ix[i] + 1;
                int indexj = jy[j] + 1;

                if (indexi >= 1 && indexi <= NGx_domain && indexj >= 1 && indexj <= NGy_domain) {
                    Ex_ion += E_x[indexi][indexj] * frazx[i] * frazy[j];
                    Ey_ion += E_y[indexi][indexj] * frazx[i] * frazy[j];
                }
            }
        }

        ions[i].vx += (qm_i * Ex_ion * (dt / 2));
        ions[i].vy += (qm_i * Ey_ion * (dt / 2));
    }

    for (int t = 0; t < NT; ++t) {

        for (int i = 0; i < NGx_domain + 2; ++i) { 
            for (int j = 0; j < NGy_domain + 2; ++j) { 
                rho[i][j] = 0.0;
                phi[i][j] = 0.0;
                E_x[i][j] = 0.0;
                E_y[i][j] = 0.0;
            }
        }

        std::vector<int> indices_to_left;
        std::vector<int> indices_to_right;

        int total_electrons_lost = 0;

        for (int i = static_cast<int>(electrons.size() - 1); i >= 0; --i) {
            electrons[i].px += electrons[i].vx * dt;
            electrons[i].py += electrons[i].vy * dt;

            if (electrons[i].px > Lx || electrons[i].px < 0 || electrons[i].py > Ly || electrons[i].py < 0) { 
                electrons.erase(electrons.begin() + i);
                total_electrons_lost += 1; 
            } 
        }

        for (int i = static_cast<int>(electrons.size() - 1); i >= 0; --i) {
            if (electrons[i].px < x0 && left != MPI_PROC_NULL) {
                indices_to_left.push_back(i); 
            }

            if (electrons[i].px >= xF && right != MPI_PROC_NULL) {
                indices_to_right.push_back(i); 
            }
        }

        std::vector<double> send_buffer_left; 
        std::vector<double> send_buffer_right;

        for (int idx : indices_to_left) {
            const Particle& e = electrons[idx];
            send_buffer_left.push_back(e.px);
            send_buffer_left.push_back(e.py);
            send_buffer_left.push_back(e.vx);
            send_buffer_left.push_back(e.vy);
        }

        for (int idx : indices_to_right) {
            const Particle& e = electrons[idx];
            send_buffer_right.push_back(e.px);
            send_buffer_right.push_back(e.py);
            send_buffer_right.push_back(e.vx);
            send_buffer_right.push_back(e.vy);
        }

        std::vector<int> indices_to_erase = indices_to_left;
        indices_to_erase.insert(indices_to_erase.end(), indices_to_right.begin(), indices_to_right.end());

        std::sort(indices_to_erase.begin(), indices_to_erase.end());
        indices_to_erase.erase(std::unique(indices_to_erase.begin(), indices_to_erase.end()), indices_to_erase.end());

        for (int i = static_cast<int>(indices_to_erase.size()) - 1; i >= 0; --i) {
            int idx = indices_to_erase[i];
            if (idx >= 0 && idx < static_cast<int>(electrons.size())) {
                electrons.erase(electrons.begin() + idx);
            }
        }

        int send_left_count = indices_to_left.size();
        int send_right_count = indices_to_right.size();

        int recv_left_count = 0;
        int recv_right_count = 0;

        MPI_Sendrecv(&send_left_count, 1, MPI_INT, left, 1, &recv_right_count, 1, MPI_INT, right, 1, cart_comm, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&send_right_count, 1, MPI_INT, right, 0, &recv_left_count, 1, MPI_INT, left, 0, cart_comm, MPI_STATUS_IGNORE);
       
        std::vector<double> recv_left_buffer(4 * recv_left_count);
        std::vector<double> recv_right_buffer(4 * recv_right_count);

        MPI_Sendrecv(send_buffer_right.data(), 4 * send_right_count, MPI_DOUBLE, right, 4, recv_left_buffer.data(), 4 * recv_left_count, MPI_DOUBLE, left, 4, cart_comm, MPI_STATUS_IGNORE);
        MPI_Sendrecv(send_buffer_left.data(), 4 * send_left_count, MPI_DOUBLE, left, 5, recv_right_buffer.data(), 4 * recv_right_count, MPI_DOUBLE, right, 5, cart_comm, MPI_STATUS_IGNORE);

        for (size_t i = 0; i < recv_left_buffer.size(); i += 4) {
            Particle e; 
            e.px = recv_left_buffer[i];
            e.py = recv_left_buffer[i + 1];
            e.vx = recv_left_buffer[i + 2];
            e.vy = recv_left_buffer[i + 3];
            electrons.push_back(e);
        }

        for (size_t i = 0; i < recv_right_buffer.size(); i += 4) {
            Particle e; 
            e.px = recv_right_buffer[i];
            e.py = recv_right_buffer[i + 1];
            e.vx = recv_right_buffer[i + 2];
            e.vy = recv_right_buffer[i + 3];
            electrons.push_back(e);
        }

        std::vector<int> indices_to_up;
        std::vector<int> indices_to_down;

        for (int i = static_cast<int>(electrons.size() - 1); i >= 0; --i) {
            if (electrons[i].py >= yF && up != MPI_PROC_NULL) {
                indices_to_up.push_back(i); 
            }

            if (electrons[i].py < y0 && down != MPI_PROC_NULL) {
                indices_to_down.push_back(i); 
            }
        }

        std::vector<double> send_buffer_up; 
        std::vector<double> send_buffer_down;

        for (int idx : indices_to_up) {
            const Particle& e = electrons[idx];
            send_buffer_up.push_back(e.px);
            send_buffer_up.push_back(e.py);
            send_buffer_up.push_back(e.vx);
            send_buffer_up.push_back(e.vy);
        }

        for (int idx : indices_to_down) {
            const Particle& e = electrons[idx];
            send_buffer_down.push_back(e.px);
            send_buffer_down.push_back(e.py);
            send_buffer_down.push_back(e.vx);
            send_buffer_down.push_back(e.vy);
        }

        indices_to_erase.clear(); 
        indices_to_erase = indices_to_up;
        indices_to_erase.insert(indices_to_erase.end(), indices_to_down.begin(), indices_to_down.end());

        std::sort(indices_to_erase.begin(), indices_to_erase.end());
        indices_to_erase.erase(std::unique(indices_to_erase.begin(), indices_to_erase.end()), indices_to_erase.end());

        for (int i = static_cast<int>(indices_to_erase.size()) - 1; i >= 0; --i) {
            int idx = indices_to_erase[i];
            if (idx >= 0 && idx < static_cast<int>(electrons.size())) {
                electrons.erase(electrons.begin() + idx);
            }
        }

        int send_up_count = indices_to_up.size(); 
        int send_down_count = indices_to_down.size();

        int recv_up_count = 0;
        int recv_down_count = 0;

        MPI_Sendrecv(&send_up_count, 1, MPI_INT, up, 6, &recv_down_count, 1, MPI_INT, down, 6, cart_comm, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&send_down_count, 1, MPI_INT, down, 7, &recv_up_count, 1, MPI_INT, up, 7, cart_comm, MPI_STATUS_IGNORE);
       
        std::vector<double> recv_up_buffer(4 * recv_up_count);
        std::vector<double> recv_down_buffer(4 * recv_down_count);

        MPI_Sendrecv(send_buffer_up.data(), 4 * send_up_count, MPI_DOUBLE, up, 8, recv_down_buffer.data(), 4 * recv_down_count, MPI_DOUBLE, down, 8, cart_comm, MPI_STATUS_IGNORE);
        MPI_Sendrecv(send_buffer_down.data(), 4 * send_down_count, MPI_DOUBLE, down, 9, recv_up_buffer.data(), 4 * recv_up_count, MPI_DOUBLE, up, 9, cart_comm, MPI_STATUS_IGNORE);

        for (size_t i = 0; i < recv_up_buffer.size(); i += 4) {
            Particle e; 
            e.px = recv_up_buffer[i];
            e.py = recv_up_buffer[i + 1];
            e.vx = recv_up_buffer[i + 2];
            e.vy = recv_up_buffer[i + 3];
            electrons.push_back(e);
        }

        for (size_t i = 0; i < recv_down_buffer.size(); i += 4) {
            Particle e; 
            e.px = recv_down_buffer[i];
            e.py = recv_down_buffer[i + 1];
            e.vx = recv_down_buffer[i + 2];
            e.vy = recv_down_buffer[i + 3];
            electrons.push_back(e);
        }

        std::vector<int> indices_to_left_ion;
        std::vector<int> indices_to_right_ion;

        int total_ions_lost = 0;

        for (int i = static_cast<int>(ions.size() - 1); i >= 0; --i) {
            ions[i].px += ions[i].vx * dt;
            ions[i].py += ions[i].vy * dt;

            if (ions[i].px > Lx || ions[i].px < 0 || ions[i].py > Ly || ions[i].py < 0) { 
                ions.erase(ions.begin() + i);
                total_ions_lost += 1; 
            } 
        }

        for (int i = static_cast<int>(ions.size() - 1); i >= 0; --i) {
            if (ions[i].px < x0 && left != MPI_PROC_NULL) {
                indices_to_left_ion.push_back(i); 
            }

            if (ions[i].px >= xF && right != MPI_PROC_NULL) {
                indices_to_right_ion.push_back(i); 
            }
        }

        std::vector<double> send_buffer_left_ion; 
        std::vector<double> send_buffer_right_ion;

        for (int idx : indices_to_left_ion) {
            const Particle& ion = ions[idx];
            send_buffer_left_ion.push_back(ion.px);
            send_buffer_left_ion.push_back(ion.py);
            send_buffer_left_ion.push_back(ion.vx);
            send_buffer_left_ion.push_back(ion.vy);
        }

        for (int idx : indices_to_right_ion) {
            const Particle& ion = ions[idx];
            send_buffer_right_ion.push_back(ion.px);
            send_buffer_right_ion.push_back(ion.py);
            send_buffer_right_ion.push_back(ion.vx);
            send_buffer_right_ion.push_back(ion.vy);
        }

        std::vector<int> indices_to_erase_ion = indices_to_left_ion;
        indices_to_erase_ion.insert(indices_to_erase_ion.end(), indices_to_right_ion.begin(), indices_to_right_ion.end());

        std::sort(indices_to_erase_ion.begin(), indices_to_erase_ion.end());
        indices_to_erase_ion.erase(std::unique(indices_to_erase_ion.begin(), indices_to_erase_ion.end()), indices_to_erase_ion.end());

        for (int i = static_cast<int>(indices_to_erase_ion.size()) - 1; i >= 0; --i) {
            int idx = indices_to_erase_ion[i];
            if (idx >= 0 && idx < static_cast<int>(ions.size())) {
                ions.erase(ions.begin() + idx);
            }
        }

        int send_left_count_ion = indices_to_left_ion.size();
        int send_right_count_ion = indices_to_right_ion.size();

        int recv_left_count_ion = 0;
        int recv_right_count_ion = 0;

        MPI_Sendrecv(&send_left_count_ion, 1, MPI_INT, left, 10, &recv_right_count_ion, 1, MPI_INT, right, 10, cart_comm, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&send_right_count_ion, 1, MPI_INT, right, 11, &recv_left_count_ion, 1, MPI_INT, left, 11, cart_comm, MPI_STATUS_IGNORE);
       
        std::vector<double> recv_left_buffer_ion(4 * recv_left_count_ion);
        std::vector<double> recv_right_buffer_ion(4 * recv_right_count_ion);

        MPI_Sendrecv(send_buffer_right_ion.data(), 4 * send_right_count_ion, MPI_DOUBLE, right, 12, recv_left_buffer_ion.data(), 4 * recv_left_count_ion, MPI_DOUBLE, left, 12, cart_comm, MPI_STATUS_IGNORE);
        MPI_Sendrecv(send_buffer_left_ion.data(), 4 * send_left_count_ion, MPI_DOUBLE, left, 13, recv_right_buffer_ion.data(), 4 * recv_right_count_ion, MPI_DOUBLE, right, 13, cart_comm, MPI_STATUS_IGNORE);

        for (size_t i = 0; i < recv_left_buffer_ion.size(); i += 4) {
            Particle ion; 
            ion.px = recv_left_buffer_ion[i];
            ion.py = recv_left_buffer_ion[i + 1];
            ion.vx = recv_left_buffer_ion[i + 2];
            ion.vy = recv_left_buffer_ion[i + 3];
            ions.push_back(ion);
        }

        for (size_t i = 0; i < recv_right_buffer_ion.size(); i += 4) {
            Particle ion; 
            ion.px = recv_right_buffer_ion[i];
            ion.py = recv_right_buffer_ion[i + 1];
            ion.vx = recv_right_buffer_ion[i + 2];
            ion.vy = recv_right_buffer_ion[i + 3];
            ions.push_back(ion);
        }

        std::vector<int> indices_to_up_ion;
        std::vector<int> indices_to_down_ion;

        for (int i = static_cast<int>(ions.size() - 1); i >= 0; --i) {
            if (ions[i].py >= yF && up != MPI_PROC_NULL) {
                    indices_to_up_ion.push_back(i); 
            }

            if (ions[i].py < y0 && down != MPI_PROC_NULL) {
                indices_to_down_ion.push_back(i); 
            }
        }
        
        std::vector<double> send_buffer_up_ion; 
        std::vector<double> send_buffer_down_ion;

        for (int idx : indices_to_up_ion) {
            const Particle& ion = ions[idx];
            send_buffer_up_ion.push_back(ion.px);
            send_buffer_up_ion.push_back(ion.py);
            send_buffer_up_ion.push_back(ion.vx);
            send_buffer_up_ion.push_back(ion.vy);
        }

        for (int idx : indices_to_down_ion) {
            const Particle& ion = ions[idx];
            send_buffer_down_ion.push_back(ion.px);
            send_buffer_down_ion.push_back(ion.py);
            send_buffer_down_ion.push_back(ion.vx);
            send_buffer_down_ion.push_back(ion.vy);
        }

        indices_to_erase_ion.clear(); 
        indices_to_erase_ion = indices_to_up_ion;
        indices_to_erase_ion.insert(indices_to_erase_ion.end(), indices_to_down_ion.begin(), indices_to_down_ion.end());

        std::sort(indices_to_erase_ion.begin(), indices_to_erase_ion.end());
        indices_to_erase_ion.erase(std::unique(indices_to_erase_ion.begin(), indices_to_erase_ion.end()), indices_to_erase_ion.end());

        for (int i = static_cast<int>(indices_to_erase_ion.size()) - 1; i >= 0; --i) {
            int idx = indices_to_erase_ion[i];
            if (idx >= 0 && idx < static_cast<int>(ions.size())) {
                ions.erase(ions.begin() + idx);
            }
        }

        int send_up_count_ion = indices_to_up_ion.size(); 
        int send_down_count_ion = indices_to_down_ion.size();

        int recv_up_count_ion = 0;
        int recv_down_count_ion = 0;

        MPI_Sendrecv(&send_up_count_ion, 1, MPI_INT, up, 14, &recv_down_count_ion, 1, MPI_INT, down, 14, cart_comm, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&send_down_count_ion, 1, MPI_INT, down, 15, &recv_up_count_ion, 1, MPI_INT, up, 15, cart_comm, MPI_STATUS_IGNORE);
       
        std::vector<double> recv_up_buffer_ion(4 * recv_up_count_ion);
        std::vector<double> recv_down_buffer_ion(4 * recv_down_count_ion);

        MPI_Sendrecv(send_buffer_up_ion.data(), 4 * send_up_count_ion, MPI_DOUBLE, up, 16, recv_down_buffer_ion.data(), 4 * recv_down_count_ion, MPI_DOUBLE, down, 16, cart_comm, MPI_STATUS_IGNORE);
        MPI_Sendrecv(send_buffer_down_ion.data(), 4 * send_down_count_ion, MPI_DOUBLE, down, 17, recv_up_buffer_ion.data(), 4 * recv_up_count_ion, MPI_DOUBLE, up, 17, cart_comm, MPI_STATUS_IGNORE);

        for (size_t i = 0; i < recv_up_buffer_ion.size(); i += 4) {
            Particle ion; 
            ion.px = recv_up_buffer_ion[i];
            ion.py = recv_up_buffer_ion[i + 1];
            ion.vx = recv_up_buffer_ion[i + 2];
            ion.vy = recv_up_buffer_ion[i + 3];
            ions.push_back(ion);
        }

        for (size_t i = 0; i < recv_down_buffer_ion.size(); i += 4) {
            Particle ion; 
            ion.px = recv_down_buffer_ion[i];
            ion.py = recv_down_buffer_ion[i + 1];
            ion.vx = recv_down_buffer_ion[i + 2];
            ion.vy = recv_down_buffer_ion[i + 3];
            ions.push_back(ion);
        }

        if (left == MPI_PROC_NULL) {
            for (int i = 0; i < total_electrons_lost; ++i) {
                Particle e; 
                e.px = 0.0; 
                e.py = uniform_y(gen); 
                e.vx = V0; 
                e.vy = 0.0;
                e.pm = 1;
                electrons.push_back(e);
            }
        }

        if (right == MPI_PROC_NULL) {
            for (int i = 0; i < total_electrons_lost; ++i) {
                Particle e; 
                e.px = Lx; 
                e.py = uniform_y(gen); 
                e.vx = -V0; 
                e.vy = 0.0;
                e.pm = -1;
                electrons.push_back(e);
            }
        }

        if (left == MPI_PROC_NULL) {
            for (int i = 0; i < total_ions_lost; ++i) {
                Particle ion; 
                ion.px = 0.0; 
                ion.py = uniform_y(gen); 
                ion.vx = V0; 
                ion.vy = 0.0;
                ion.pm = 1;
                ions.push_back(ion);
            }
        }

        if (right == MPI_PROC_NULL) {
            for (int i = 0; i < total_ions_lost; ++i) {
                Particle ion; 
                ion.px = Lx; 
                ion.py = uniform_y(gen); 
                ion.vx = -V0; 
                ion.vy = 0.0;
                ion.pm = -1;
                ions.push_back(ion);
            }
        }

        for (int i = static_cast<int>(electrons.size()) - 1; i >= 0; --i) {
            double dx = electrons[i].px - sphere.x;
            double dy = electrons[i].py - sphere.y;
            double dist = std::sqrt(dx*dx + dy*dy);

            if (dist <= sphere.radius) {
                sphere.charge += Q_e;
                electrons.erase(electrons.begin() + i);
            }
        }

        for (int i = static_cast<int>(ions.size()) - 1; i >= 0; --i) {
            double dx = ions[i].px - sphere.x;
            double dy = ions[i].py - sphere.y;
            double dist = std::sqrt(dx*dx + dy*dy);

            if (dist <= sphere.radius) {
                sphere.charge += Q_i;
                ions.erase(ions.begin() + i);
            }
        }

        double local_charge = sphere.charge;
        double total_charge = 0.0;

        MPI_Reduce(&local_charge, &total_charge, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Bcast(&total_charge, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        std::vector<double> Ex_sphere_electrons(electrons.size(), 0.0);
        std::vector<double> Ey_sphere_electrons(electrons.size(), 0.0);

        for (size_t i = 0; i < electrons.size(); ++i) {
            double dx = electrons[i].px - sphere.x;
            double dy = electrons[i].py - sphere.y;
            double r2 = dx * dx + dy * dy;
            double r = std::sqrt(r2);

            if (r > sphere.radius && r > 1e-12) {
                double factor = k * total_charge / (r2 * r);  // k = 1 / (4*pi*eps0)
                Ex_sphere_electrons[i] = factor * dx;
                Ey_sphere_electrons[i] = factor * dy;
            }
        }

        std::vector<double> Ex_sphere_ions(ions.size(), 0.0);
        std::vector<double> Ey_sphere_ions(ions.size(), 0.0);

        for (size_t i = 0; i < ions.size(); ++i) {
            double dx = ions[i].px - sphere.x;
            double dy = ions[i].py - sphere.y;
            double r2 = dx * dx + dy * dy;
            double r = std::sqrt(r2);

            if (r > sphere.radius && r > 1e-12) {
                double factor = k * total_charge / (r2 * r);  // k = 1 / (4*pi*eps0)
                Ex_sphere_ions[i] = factor * dx;
                Ey_sphere_ions[i] = factor * dy;
            }
        }

        for (int i = 0; i < static_cast<int>(electrons.size()); ++i) {
            double px_local = electrons[i].px - x0;
            double py_local = electrons[i].py - y0;

            double px_grid = px_local / dx;
            double py_grid = py_local / dy;

            int i1 = static_cast<int>(std::floor(px_grid - 0.5));
            int j1 = static_cast<int>(std::floor(py_grid - 0.5));

            i1 = std::max(0, std::min(i1, NGx_domain - 2));
            j1 = std::max(0, std::min(j1, NGy_domain - 2));

            std::array<int, 2> ix = {i1, i1 + 1};
            std::array<int, 2> jy = {j1, j1 + 1};

            double frazi1 = px_grid - i1 - 0.5;
            double frazj1 = py_grid - j1 - 0.5;

            frazi1 = std::min(std::max(frazi1, 0.0), 1.0);
            frazj1 = std::min(std::max(frazj1, 0.0), 1.0);

            double w_ileft = 1 - frazi1;
            double w_iright = frazi1; 
            double w_jdown = 1 - frazj1; 
            double w_jup = frazj1;

            std::array<double, 2> frazx = {w_ileft, w_iright};
            std::array<double, 2> frazy = {w_jdown, w_jup};

            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 2; ++j) {
                    int indexi = ix[i] + 1; 
                    int indexj = jy[j] + 1; 
                    if (indexi <= NGx_domain && indexi >= 1 && indexj <= NGy_domain && indexj >=1) {
                        rho[indexi][indexj] = rho[indexi][indexj] + ((Q_e / (dx * dy)) * frazx[i] * frazy[j]);
                    }
                }
            }
        }

        for (int i = 0; i < static_cast<int>(ions.size()); ++i) {
            double px_local = ions[i].px - x0;
            double py_local = ions[i].py - y0;

            double px_grid = px_local / dx;
            double py_grid = py_local / dy;

            int i1 = static_cast<int>(std::floor(px_grid - 0.5));
            int j1 = static_cast<int>(std::floor(py_grid - 0.5));

            i1 = std::max(0, std::min(i1, NGx_domain - 2));
            j1 = std::max(0, std::min(j1, NGy_domain - 2));

            std::array<int, 2> ix = {i1, i1 + 1};
            std::array<int, 2> jy = {j1, j1 + 1};

            double frazi1 = px_grid - i1 - 0.5;
            double frazj1 = py_grid - j1 - 0.5;

            frazi1 = std::min(std::max(frazi1, 0.0), 1.0);
            frazj1 = std::min(std::max(frazj1, 0.0), 1.0);

            double w_ileft = 1 - frazi1;
            double w_iright = frazi1; 
            double w_jdown = 1 - frazj1; 
            double w_jup = frazj1;

            std::array<double, 2> frazx = {w_ileft, w_iright};
            std::array<double, 2> frazy = {w_jdown, w_jup};

            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 2; ++j) {
                    int indexi = ix[i] + 1; 
                    int indexj = jy[j] + 1; 
                    if (indexi <= NGx_domain && indexi >= 1 && indexj <= NGy_domain && indexj >=1) {
                        rho[indexi][indexj] = rho[indexi][indexj] + ((Q_i / (dx * dy)) * frazx[i] * frazy[j]);
                    }
                }
            }
        }

        std::vector<double> send_buffer_left_grid(NGy_domain); 
        std::vector<double> send_buffer_right_grid(NGy_domain); 
        std::vector<double> send_buffer_up_grid(NGx_domain); 
        std::vector<double> send_buffer_down_grid(NGx_domain); 

        std::vector<double> recv_buffer_left_grid(NGy_domain); 
        std::vector<double> recv_buffer_right_grid(NGy_domain);
        std::vector<double> recv_buffer_up_grid(NGx_domain); 
        std::vector<double> recv_buffer_down_grid(NGx_domain);

        if (left != MPI_PROC_NULL) {
            for (int j = 1; j <= NGy_domain; ++j) { 
                send_buffer_left_grid[j - 1] = rho[1][j];
            }

            MPI_Sendrecv(send_buffer_left_grid.data(), NGy_domain, MPI_DOUBLE, left, 0, recv_buffer_right_grid.data(), NGy_domain, MPI_DOUBLE, right, 0, cart_comm, MPI_STATUS_IGNORE);
        }

        if (right != MPI_PROC_NULL) {
            for (int j = 1; j <= NGy_domain; ++j) {
                send_buffer_right_grid[j - 1] = rho[NGx_domain][j];
            }

            MPI_Sendrecv(send_buffer_right_grid.data(), NGy_domain, MPI_DOUBLE, right, 1, recv_buffer_left_grid.data(), NGy_domain, MPI_DOUBLE, left, 1, cart_comm, MPI_STATUS_IGNORE);
        }

        if (up != MPI_PROC_NULL) {
            for (int i = 1; i <= NGx_domain; ++i) {
                send_buffer_up_grid[i - 1] = rho[i][NGy_domain];
            }

            MPI_Sendrecv(send_buffer_up_grid.data(), NGx_domain, MPI_DOUBLE, up, 2, recv_buffer_down_grid.data(), NGx_domain, MPI_DOUBLE, down, 2, cart_comm, MPI_STATUS_IGNORE);
        }

        if (down != MPI_PROC_NULL) {
            for (int i = 1; i <= NGx_domain; ++i) {
                send_buffer_down_grid[i - 1] = rho[i][1]; 
            }

            MPI_Sendrecv(send_buffer_down_grid.data(), NGx_domain, MPI_DOUBLE, down, 3, recv_buffer_up_grid.data(), NGx_domain, MPI_DOUBLE, up, 3, cart_comm, MPI_STATUS_IGNORE);
    
        }

        if (left != MPI_PROC_NULL) {
            for (int j = 1; j <= NGy_domain; ++j) {
                rho[0][j] = recv_buffer_left_grid[j - 1];
            }
        }

        if (right != MPI_PROC_NULL) {
            for (int j = 1; j <= NGy_domain; ++j) {
                rho[NGx_domain + 1][j] = recv_buffer_right_grid[j - 1];
            }
        }

        if (up != MPI_PROC_NULL) {
            for (int i = 1; i <= NGx_domain; ++i) {
                rho[i][NGy_domain + 1] = recv_buffer_up_grid_initial[i - 1];
            }
        }

        if (down != MPI_PROC_NULL) {
            for (int i = 1; i <= NGx_domain; ++i) {
                rho[i][0] = recv_buffer_down_grid_initial[i - 1];
            }
        }

        Poisson_solver(phi, rho, NGx_domain, NGy_domain, dx, dy, left, right, up, down, cart_comm);

        Field_solver(phi, E_x, E_y, NGx_domain, NGy_domain, dx, dy);

        for (int i = 0; i < static_cast<int>(electrons.size()); ++i) {
            double px_local = electrons[i].px - x0;
            double py_local = electrons[i].py - y0;

            double px_grid = px_local / dx;
            double py_grid = py_local / dy;

            int i1 = static_cast<int>(std::floor(px_grid - 0.5));
            int j1 = static_cast<int>(std::floor(py_grid - 0.5));

            std::array<int, 2> ix = {i1, i1 + 1};
            std::array<int, 2> jy = {j1, j1 + 1};

            double frazi1 = px_grid - i1 - 0.5;
            double frazj1 = py_grid - j1 - 0.5;

            double w_ileft = 1 - frazi1;
            double w_iright = frazi1; 
            double w_jdown = 1 - frazj1; 
            double w_jup = frazj1; 

            std::array<double, 2> frazx = {w_ileft, w_iright};
            std::array<double, 2> frazy = {w_jdown, w_jup};

            double Ex_electron = 0;
            double Ey_electron = 0;

            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 2; ++j) {
                    int indexi = ix[i] + 1;
                    int indexj = jy[j] + 1;

                    if (indexi >= 1 && indexi <= NGx_domain && indexj >= 1 && indexj <= NGy_domain) {
                        Ex_electron += E_x[indexi][indexj] * frazx[i] * frazy[j];
                        Ey_electron += E_y[indexi][indexj] * frazx[i] * frazy[j];
                    }
                }
            }

            Ex_electron += Ex_sphere_electrons[i];
            Ey_electron += Ey_sphere_electrons[i];

            electrons[i].vx += (qm_e * Ex_electron * dt);
            electrons[i].vy += (qm_e * Ey_electron * dt);
        }

        for (int i = 0; i < static_cast<int>(ions.size()); ++i) {
            double px_local = ions[i].px - x0;
            double py_local = ions[i].py - y0;

            double px_grid = px_local / dx;
            double py_grid = py_local / dy;

            int i1 = static_cast<int>(std::floor(px_grid - 0.5));
            int j1 = static_cast<int>(std::floor(py_grid - 0.5));

            std::array<int, 2> ix = {i1, i1 + 1};
            std::array<int, 2> jy = {j1, j1 + 1};

            double frazi1 = px_grid - i1 - 0.5;
            double frazj1 = py_grid - j1 - 0.5;

            double w_ileft = 1 - frazi1;
            double w_iright = frazi1; 
            double w_jdown = 1 - frazj1; 
            double w_jup = frazj1; 

            std::array<double, 2> frazx = {w_ileft, w_iright};
            std::array<double, 2> frazy = {w_jdown, w_jup};

            double Ex_ion = 0.0;
            double Ey_ion = 0.0;

            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 2; ++j) {
                    int indexi = ix[i] + 1;
                    int indexj = jy[j] + 1;

                    if (indexi >= 1 && indexi <= NGx_domain && indexj >= 1 && indexj <= NGy_domain) {
                        Ex_ion += E_x[indexi][indexj] * frazx[i] * frazy[j];
                        Ey_ion += E_y[indexi][indexj] * frazx[i] * frazy[j];
                    }
                }
            }

            Ex_ion += Ex_sphere_ions[i];
            Ey_ion += Ey_sphere_ions[i]; 

            ions[i].vx += (qm_i * Ex_ion * dt);
            ions[i].vy += (qm_i * Ey_ion * dt);
        }

        std::cout << "step " << t << " done" << std::endl;
    }

    double local_charge = sphere.charge;
    double total_charge = 0.0;

    MPI_Reduce(&local_charge, &total_charge, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank ==0 ) std::cout << "Total sphere charge; " << total_charge << std::endl;

    MPI_Finalize(); 
    return 0;
}
// mpic++ -O2 -std=c++17 iPIC2D_MPI_2.0.cpp -o iPIC2D_MPI_2.0