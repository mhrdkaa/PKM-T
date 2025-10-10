-- phpMyAdmin SQL Dump
-- version 5.2.1
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Generation Time: Sep 14, 2025 at 12:16 PM
-- Server version: 10.4.28-MariaDB
-- PHP Version: 8.0.28

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `pkm_db`
--

-- --------------------------------------------------------

--
-- Table structure for table `resi_db`
--

CREATE TABLE `resi_db` (
  `id` int(11) NOT NULL,
  `resi` varchar(255) NOT NULL,
  `barang` varchar(255) NOT NULL,
  `harga` varchar(255) NOT NULL,
  `time` timestamp NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `resi_db`
--

INSERT INTO `resi_db` (`id`, `resi`, `barang`, `harga`, `time`) VALUES
(1, '1234', '', '', '2025-08-24 11:39:34'),
(2, '1234', '', '', '2025-08-24 11:39:34'),
(3, '12345700', '', '', '2025-08-24 11:39:34'),
(4, '11111', '', '', '2025-08-24 11:39:34'),
(5, '2147480000', '', '', '2025-08-24 11:39:34'),
(6, '100000000000', '', '', '2025-08-24 11:39:34'),
(7, '6922794772115', '', '', '2025-08-24 11:39:34'),
(8, '692279477211', '', '', '2025-08-24 11:39:34'),
(9, '692279477211', '', '', '2025-08-24 11:39:34'),
(11, '69227947721134', '', '', '2025-08-24 11:39:34'),
(12, '692279477211345', '', '', '2025-08-24 11:39:34'),
(13, '123456', 'emasss', '12343', '2025-08-24 12:12:13'),
(14, '1111111', 'sepatu', '150000', '2025-08-24 12:13:44'),
(15, '9785507438', 'webcam', '200000', '2025-08-24 12:49:11'),
(16, '6922794772113', 'webcam', '350000', '2025-08-24 12:52:54'),
(17, '765756931243', 'Raspi', '300000', '2025-09-14 09:06:52');

--
-- Indexes for dumped tables
--

--
-- Indexes for table `resi_db`
--
ALTER TABLE `resi_db`
  ADD PRIMARY KEY (`id`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `resi_db`
--
ALTER TABLE `resi_db`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=18;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
