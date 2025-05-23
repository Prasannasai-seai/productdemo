import React, { useState } from 'react';
import {
  Box,
  Button,
  FormControl,
  FormLabel,
  Input,
  VStack,
  Text,
  Alert,
  AlertIcon,
  useToast,
  Spinner,
  Card,
  CardBody,
  CardHeader,
  Heading,
  InputGroup,
  InputLeftAddon,
  Divider
} from '@chakra-ui/react';
import axios from 'axios';

interface TrainingFormProps {
  onTrainingComplete?: (result: any) => void;
}

interface TrainingResult {
  status: string;
  r2_score?: number;
  mae?: number;
  mse?: number;
  message?: string;
}

const TrainingForm: React.FC<TrainingFormProps> = ({ onTrainingComplete }) => {
  const [placeTable, setPlaceTable] = useState('');
  const [ctsTable, setCtsTable] = useState('');
  const [routeTable, setRouteTable] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<TrainingResult | null>(null);
  
  const toast = useToast();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await axios.post('/slack-prediction/train', {
        place_table: placeTable,
        cts_table: ctsTable,
        route_table: routeTable
      }, {
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        validateStatus: function (status) {
          return status >= 200 && status < 500; // Accept any status code to handle errors properly
        }
      });

      if (response.status === 200 && response.data.status === 'success') {
        setResult(response.data);
        toast({
          title: 'Training Complete',
          description: 'Model training completed successfully!',
          status: 'success',
          duration: 5000,
          isClosable: true,
        });
        
        if (onTrainingComplete) {
          onTrainingComplete(response.data);
        }
      } else {
        // Handle error response
        const errorMessage = response.data?.message || response.data?.detail?.message || 'Training failed';
        throw new Error(errorMessage);
      }
    } catch (err) {
      console.error('Training error:', err);
      let errorMessage = 'An error occurred during training';
      
      if (axios.isAxiosError(err)) {
        if (err.response) {
          // The server responded with an error
          const responseData = err.response.data;
          errorMessage = responseData?.detail?.message || responseData?.message || responseData?.detail || err.message;
        } else if (err.request) {
          // The request was made but no response was received
          errorMessage = 'No response received from server. Please check if the server is running.';
        } else {
          // Something happened in setting up the request
          errorMessage = err.message;
        }
      } else if (err instanceof Error) {
        errorMessage = err.message;
      }

      setError(errorMessage);
      toast({
        title: 'Training Failed',
        description: errorMessage,
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleQuickFill = () => {
    setPlaceTable('ariane_place_sorted');
    setCtsTable('ariane_cts_sorted');
    setRouteTable('ariane_route_sorted');
  };

  return (
    <Card variant="outline" bg="white" boxShadow="sm">
      <CardHeader pb={2}>
        <Heading size="md" color="gray.700">Model Training</Heading>
      </CardHeader>
      <CardBody>
        <form onSubmit={handleSubmit}>
          <VStack spacing={4} align="stretch">
            <Text fontSize="sm" color="gray.600" mb={2}>
              Enter the table names for training the model or use the Quick Fill button for example tables.
            </Text>

            <FormControl isRequired>
              <FormLabel fontSize="sm" color="gray.700">Place Table</FormLabel>
              <InputGroup>
                <Input
                  value={placeTable}
                  onChange={(e) => setPlaceTable(e.target.value)}
                  placeholder="e.g., ariane_place_sorted"
                  isDisabled={isLoading}
                  bg="white"
                  borderColor="gray.300"
                  _hover={{ borderColor: 'gray.400' }}
                  _focus={{ borderColor: 'blue.500', boxShadow: 'outline' }}
                />
              </InputGroup>
            </FormControl>

            <FormControl isRequired>
              <FormLabel fontSize="sm" color="gray.700">CTS Table</FormLabel>
              <InputGroup>
                <Input
                  value={ctsTable}
                  onChange={(e) => setCtsTable(e.target.value)}
                  placeholder="e.g., ariane_cts_sorted"
                  isDisabled={isLoading}
                  bg="white"
                  borderColor="gray.300"
                  _hover={{ borderColor: 'gray.400' }}
                  _focus={{ borderColor: 'blue.500', boxShadow: 'outline' }}
                />
              </InputGroup>
            </FormControl>

            <FormControl isRequired>
              <FormLabel fontSize="sm" color="gray.700">Route Table</FormLabel>
              <InputGroup>
                <Input
                  value={routeTable}
                  onChange={(e) => setRouteTable(e.target.value)}
                  placeholder="e.g., ariane_route_sorted"
                  isDisabled={isLoading}
                  bg="white"
                  borderColor="gray.300"
                  _hover={{ borderColor: 'gray.400' }}
                  _focus={{ borderColor: 'blue.500', boxShadow: 'outline' }}
                />
              </InputGroup>
            </FormControl>

            <Divider my={2} />

            <VStack spacing={3}>
              <Button
                w="full"
                colorScheme="gray"
                variant="outline"
                onClick={handleQuickFill}
                size="sm"
                isDisabled={isLoading}
                leftIcon={<span>📋</span>}
              >
                Quick Fill Example Tables
              </Button>

              <Button
                w="full"
                type="submit"
                colorScheme="blue"
                isLoading={isLoading}
                loadingText="Training..."
                isDisabled={!placeTable || !ctsTable || !routeTable}
                leftIcon={isLoading ? <Spinner size="sm" /> : <span>🚀</span>}
              >
                Start Training
              </Button>
            </VStack>
          </VStack>
        </form>

        {error && (
          <Alert status="error" mt={4} borderRadius="md">
            <AlertIcon />
            <Text fontSize="sm">{error}</Text>
          </Alert>
        )}

        {result && result.status === 'success' && (
          <Alert status="success" mt={4} borderRadius="md">
            <AlertIcon />
            <VStack align="stretch" spacing={1}>
              <Text fontWeight="bold">Training completed successfully!</Text>
              <Text fontSize="sm">R² Score: {result.r2_score?.toFixed(4)}</Text>
              <Text fontSize="sm">Mean Absolute Error: {result.mae?.toFixed(4)}</Text>
              <Text fontSize="sm">Mean Squared Error: {result.mse?.toFixed(4)}</Text>
            </VStack>
          </Alert>
        )}
      </CardBody>
    </Card>
  );
};

export default TrainingForm; 